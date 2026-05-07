#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  alignas(64) float x[N], y[N], m[N], fx[N], fy[N];

  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  __m512 xvec = _mm512_load_ps(x);
  __m512 yvec = _mm512_load_ps(y);
  __m512 mvec = _mm512_load_ps(m);
  __m512 one = _mm512_set1_ps(1.0f);

  for(int i=0; i<N; i++) {
    __m512 xi = _mm512_set1_ps(x[i]);
    __m512 yi = _mm512_set1_ps(y[i]);

    __m512 rx = _mm512_sub_ps(xi, xvec);
    __m512 ry = _mm512_sub_ps(yi, yvec);

    __m512 r2 = _mm512_add_ps(
      _mm512_mul_ps(rx, rx),
      _mm512_mul_ps(ry, ry)
    );

    __mmask16 mask = ~(1 << i);

    r2 = _mm512_mask_blend_ps(mask, one, r2);

    __m512 inv_r = _mm512_rsqrt14_ps(r2);
    __m512 inv_r3 = _mm512_mul_ps(
      _mm512_mul_ps(inv_r, inv_r),
      inv_r
    );

    __m512 coef = _mm512_mul_ps(mvec, inv_r3);

    __m512 dfx = _mm512_mul_ps(rx, coef);
    __m512 dfy = _mm512_mul_ps(ry, coef);

    fx[i] -= _mm512_mask_reduce_add_ps(mask, dfx);
    fy[i] -= _mm512_mask_reduce_add_ps(mask, dfy);

    printf("%d %g %g\n", i, fx[i], fy[i]);
  }
}