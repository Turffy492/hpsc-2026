#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline_primitives.h>

using namespace std;
using namespace nvcuda;

namespace tc {
enum {
  tm = 128,
  tn = 128,
  tk = 64,
  wm = 64,
  wn = 64,
  stages = 3,
  pad_a = 8,
  pad_b = 8,
  group_m = 8,
  mma_m = 16,
  mma_n = 16,
  mma_k = 16
};

enum {
  warps_m = tm / wm,
  warps_n = tn / wn,
  warps = warps_m * warps_n,
  threads = warps * 32,
  ld_a = tm + pad_a,
  ld_b = tk + pad_b,
  frag_m = wm / mma_m,
  frag_n = wn / mma_n
};

static_assert(tm % 8 == 0, "");
static_assert(tn % 8 == 0, "");
static_assert(tk % 16 == 0, "");
static_assert(ld_a % 8 == 0, "");
static_assert(ld_b % 8 == 0, "");
static_assert(warps == 4, "");
}

__global__ void float_to_half_kernel(int n, const float *x, half *y) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int step = blockDim.x * gridDim.x;
  for (int i = id; i < n; i += step) y[i] = __float2half_rn(x[i]);
}

__device__ __forceinline__ void copy16(half *dst, const half *src) {
  __pipeline_memcpy_async(reinterpret_cast<int4*>(dst),
                          reinterpret_cast<const int4*>(src),
                          sizeof(int4));
}

__device__ __forceinline__ void load_ab_tile(const half *a,
                                             const half *b,
                                             half *sa,
                                             half *sb,
                                             int dim_m,
                                             int dim_k,
                                             int m0,
                                             int n0,
                                             int k0,
                                             int buf) {
  half *ta = sa + buf * tc::tk * tc::ld_a;
  half *tb = sb + buf * tc::tn * tc::ld_b;

#pragma unroll
  for (int v = threadIdx.x; v < (tc::tm * tc::tk) / 8; v += tc::threads) {
    int lm = (v % (tc::tm / 8)) * 8;
    int lk = v / (tc::tm / 8);
    copy16(ta + lk * tc::ld_a + lm, a + (k0 + lk) * dim_m + m0 + lm);
  }

#pragma unroll
  for (int v = threadIdx.x; v < (tc::tn * tc::tk) / 8; v += tc::threads) {
    int lk = (v % (tc::tk / 8)) * 8;
    int ln = v / (tc::tk / 8);
    copy16(tb + ln * tc::ld_b + lk, b + (n0 + ln) * dim_k + k0 + lk);
  }

  __pipeline_commit();
}

using AF = wmma::fragment<wmma::matrix_a, tc::mma_m, tc::mma_n, tc::mma_k, half, wmma::col_major>;
using BF = wmma::fragment<wmma::matrix_b, tc::mma_m, tc::mma_n, tc::mma_k, half, wmma::col_major>;
using CF = wmma::fragment<wmma::accumulator, tc::mma_m, tc::mma_n, tc::mma_k, float>;

__device__ __forceinline__ void compute_ab_tile(const half *sa,
                                                const half *sb,
                                                int warp_m0,
                                                int warp_n0,
                                                CF (&c)[tc::frag_m][tc::frag_n]) {
  AF af[tc::frag_m];
  BF bf[tc::frag_n];

#pragma unroll
  for (int kk = 0; kk < tc::tk; kk += tc::mma_k) {
#pragma unroll
    for (int i = 0; i < tc::frag_m; ++i) {
      wmma::load_matrix_sync(af[i],
                             sa + kk * tc::ld_a + warp_m0 + i * tc::mma_m,
                             tc::ld_a);
    }

#pragma unroll
    for (int j = 0; j < tc::frag_n; ++j) {
      wmma::load_matrix_sync(bf[j],
                             sb + (warp_n0 + j * tc::mma_n) * tc::ld_b + kk,
                             tc::ld_b);
    }

#pragma unroll
    for (int i = 0; i < tc::frag_m; ++i) {
#pragma unroll
      for (int j = 0; j < tc::frag_n; ++j) {
        wmma::mma_sync(c[i][j], af[i], bf[j], c[i][j]);
      }
    }
  }
}

__global__ __launch_bounds__(tc::threads, 2)
void tensorcore_kernel(int dim_m, int dim_n, int dim_k,
                       const half *__restrict__ a,
                       const half *__restrict__ b,
                       float *__restrict__ c) {
  extern __shared__ half smem[];
  half *sa = smem;
  half *sb = sa + tc::stages * tc::tk * tc::ld_a;

  int nb_m = gridDim.x;
  int nb_n = gridDim.y;
  int linear = blockIdx.x + blockIdx.y * nb_m;
  int group = tc::group_m * nb_n;
  int base_m = (linear / group) * tc::group_m;
  int gm = nb_m - base_m;
  if (gm > tc::group_m) gm = tc::group_m;
  int local = linear - (linear / group) * group;
  int tile_n = local / gm;
  int tile_m = base_m + local - tile_n * gm;

  int block_m = tile_m * tc::tm;
  int block_n = tile_n * tc::tn;

  int warp = threadIdx.x >> 5;
  int warp_m = (warp / tc::warps_n) * tc::wm;
  int warp_n = (warp % tc::warps_n) * tc::wn;

  CF acc[tc::frag_m][tc::frag_n];
#pragma unroll
  for (int i = 0; i < tc::frag_m; ++i) {
#pragma unroll
    for (int j = 0; j < tc::frag_n; ++j) {
      wmma::fill_fragment(acc[i][j], 0.0f);
    }
  }

  int nk = dim_k / tc::tk;
  int first = tc::stages - 1;
  if (first > nk) first = nk;

#pragma unroll
  for (int s = 0; s < tc::stages - 1; ++s) {
    if (s < first) {
      load_ab_tile(a, b, sa, sb, dim_m, dim_k,
                   block_m, block_n, s * tc::tk, s);
    }
  }

  int next = first;
  int read = 0;

#pragma unroll 1
  for (int t = 0; t < nk; ++t) {
    if (next < nk) {
      int write = next % tc::stages;
      load_ab_tile(a, b, sa, sb, dim_m, dim_k,
                   block_m, block_n, next * tc::tk, write);
      __pipeline_wait_prior(tc::stages - 2);
    } else {
      __pipeline_wait_prior(0);
    }

    __syncthreads();

    const half *ta = sa + read * tc::tk * tc::ld_a;
    const half *tb = sb + read * tc::tn * tc::ld_b;
    compute_ab_tile(ta, tb, warp_m, warp_n, acc);

    __syncthreads();

    read = (read + 1) % tc::stages;
    ++next;
  }

#pragma unroll
  for (int i = 0; i < tc::frag_m; ++i) {
#pragma unroll
    for (int j = 0; j < tc::frag_n; ++j) {
      int cm = block_m + warp_m + i * tc::mma_m;
      int cn = block_n + warp_n + j * tc::mma_n;
      wmma::store_matrix_sync(c + cn * dim_m + cm,
                              acc[i][j],
                              dim_m,
                              wmma::mem_col_major);
    }
  }
}

int main(int argc, const char **argv) {
  int m = 10240;
  int k = 4096;
  int n = 8192;
  float alpha = 1.0;
  float beta = 0.0;
  int Nt = 10;

  if (m % tc::tm != 0 || n % tc::tn != 0 || k % tc::tk != 0) {
    fprintf(stderr, "matrix size is not compatible with the selected tile size\n");
    return 1;
  }

  float *A, *B, *C, *C2;
  half *A16, *B16;

  cudaMallocManaged(&A, size_t(m) * k * sizeof(float));
  cudaMallocManaged(&B, size_t(k) * n * sizeof(float));
  cudaMallocManaged(&C, size_t(m) * n * sizeof(float));
  cudaMallocManaged(&C2, size_t(m) * n * sizeof(float));
  cudaMallocManaged(&A16, size_t(m) * k * sizeof(half));
  cudaMallocManaged(&B16, size_t(k) * n * sizeof(half));

  srand48(1);
  for (int i = 0; i < m; i++)
    for (int j = 0; j < k; j++)
      A[size_t(k) * i + j] = drand48();

  for (int i = 0; i < k; i++)
    for (int j = 0; j < n; j++)
      B[size_t(n) * i + j] = drand48();

  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      C[size_t(m) * i + j] = C2[size_t(m) * i + j] = 0.0f;

  int device = 0;
  cudaGetDevice(&device);
  cudaMemPrefetchAsync(A, size_t(m) * k * sizeof(float), device);
  cudaMemPrefetchAsync(B, size_t(k) * n * sizeof(float), device);
  cudaMemPrefetchAsync(C, size_t(m) * n * sizeof(float), device);
  cudaMemPrefetchAsync(C2, size_t(m) * n * sizeof(float), device);
  cudaMemPrefetchAsync(A16, size_t(m) * k * sizeof(half), device);
  cudaMemPrefetchAsync(B16, size_t(k) * n * sizeof(half), device);
  cudaDeviceSynchronize();

  float_to_half_kernel<<<4096, 256>>>(m * k, A, A16);
  float_to_half_kernel<<<4096, 256>>>(k * n, B, B16);
  cudaDeviceSynchronize();

  cublasHandle_t cublas_handle;
  cublasCreate(&cublas_handle);
  cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH);

  auto tic = chrono::steady_clock::now();
  for (int i = 0; i < Nt + 2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    cublasGemmEx(cublas_handle,
                 CUBLAS_OP_N,
                 CUBLAS_OP_N,
                 m,
                 n,
                 k,
                 &alpha,
                 A,
                 CUDA_R_32F,
                 m,
                 B,
                 CUDA_R_32F,
                 k,
                 &beta,
                 C,
                 CUDA_R_32F,
                 m,
                 CUBLAS_COMPUTE_32F_FAST_16F,
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
  }

  auto toc = chrono::steady_clock::now();
  int64_t num_flops =
      (2 * int64_t(m) * int64_t(n) * int64_t(k)) +
      (2 * int64_t(m) * int64_t(n));
  double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
  double cublas_flops = double(num_flops) / tcublas / 1.0e9;

  dim3 block(tc::threads);
  dim3 grid(m / tc::tm, n / tc::tn);
  size_t smem_size =
      tc::stages * tc::tk * tc::ld_a * sizeof(half) +
      tc::stages * tc::tn * tc::ld_b * sizeof(half);

  cudaFuncSetAttribute(tensorcore_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       static_cast<int>(smem_size));
  cudaFuncSetCacheConfig(tensorcore_kernel, cudaFuncCachePreferShared);

  for (int i = 0; i < Nt + 2; i++) {
    if (i == 2) tic = chrono::steady_clock::now();
    tensorcore_kernel<<<grid, block, smem_size>>>(m, n, k, A16, B16, C2);
    cudaDeviceSynchronize();
  }

  toc = chrono::steady_clock::now();
  double tcutlass = chrono::duration<double>(toc - tic).count() / Nt;
  double cutlass_flops = double(num_flops) / tcutlass / 1.0e9;

  printf("CUBLAS: %.2f Gflops, CUTLASS: %.2f Gflops\n", cublas_flops, cutlass_flops);

  cudaMemPrefetchAsync(C, size_t(m) * n * sizeof(float), cudaCpuDeviceId);
  cudaMemPrefetchAsync(C2, size_t(m) * n * sizeof(float), cudaCpuDeviceId);
  cudaDeviceSynchronize();

  double err = 0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < m; j++)
      err += fabs(double(C[size_t(m) * i + j]) -
                  double(C2[size_t(m) * i + j]));

  printf("error: %lf\n", err / double(n) / double(m));

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
  cudaFree(C2);
  cudaFree(A16);
  cudaFree(B16);
  cublasDestroy(cublas_handle);
  return 0;
}
