#include "vg_util.h"
#include <vg/vg.h>
#include <bx/bx.h>

#if VG_CONFIG_ENABLE_SIMD && BX_CPU_X86
#include <xmmintrin.h>
#include <immintrin.h>
#endif

BX_PRAGMA_DIAGNOSTIC_IGNORED_GCC("-Wimplicit-fallthrough=0")

namespace vgutil
{
bool invertMatrix3(const float* __restrict t, float* __restrict inv)
{
	// nvgTransformInverse
	double invdet, det = (double)t[0] * t[3] - (double)t[2] * t[1];
	if (det > -1e-6 && det < 1e-6) {
		inv[0] = inv[3] = 1.0f;
		inv[1] = inv[2] = inv[4] = inv[5] = 0.0f;
		return false;
	}

	invdet = 1.0 / det;
	inv[0] = (float)(t[3] * invdet);
	inv[2] = (float)(-t[2] * invdet);
	inv[4] = (float)(((double)t[2] * t[5] - (double)t[3] * t[4]) * invdet);
	inv[1] = (float)(-t[1] * invdet);
	inv[3] = (float)(t[0] * invdet);
	inv[5] = (float)(((double)t[1] * t[4] - (double)t[0] * t[5]) * invdet);

	return true;
}

#if VG_CONFIG_ENABLE_SIMD && BX_CPU_X86
void memset32(void* __restrict dst, uint32_t n, const void* __restrict src)
{
	const __m128 s128 = _mm_load_ps1((const float*)src);
	float* d = (float*)dst;

	uint32_t iter = n >> 4;
	while (iter-- > 0) {
		_mm_storeu_ps(d + 0, s128);
		_mm_storeu_ps(d + 4, s128);
		_mm_storeu_ps(d + 8, s128);
		_mm_storeu_ps(d + 12, s128);
		d += 16;
	}

	uint32_t rem = n & 15;
	if (rem >= 8) {
		_mm_storeu_ps(d + 0, s128);
		_mm_storeu_ps(d + 4, s128);
		d += 8;
		rem -= 8;
	}

	if (rem >= 4) {
		_mm_storeu_ps(d, s128);
		d += 4;
		rem -= 4;
	}

	switch (rem) {
	case 3: *d++ = *(const float*)src;
	case 2: *d++ = *(const float*)src;
	case 1: *d = *(const float*)src;
	}
}

void memset64(void* __restrict dst, uint32_t n64, const void* __restrict src)
{
	const float* s = (const float*)src;
	const __m128 s0 = _mm_load_ss(&s[0]);
	const __m128 s1 = _mm_load_ss(&s[1]);
	const __m128 s0011 = _mm_shuffle_ps(s0, s1, _MM_SHUFFLE(0, 0, 0, 0));
	const __m128 s128 = _mm_shuffle_ps(s0011, s0011, _MM_SHUFFLE(2, 0, 2, 0));
	float* d = (float*)dst;

	uint32_t iter = n64 >> 3; // 8 64-bit values per iteration (== 16 floats / iter)
	while (iter-- > 0) {
		_mm_storeu_ps(d + 0, s128);
		_mm_storeu_ps(d + 4, s128);
		_mm_storeu_ps(d + 8, s128);
		_mm_storeu_ps(d + 12, s128);
		d += 16;
	}

	uint32_t rem = n64 & 7;
	if (rem >= 4) {
		_mm_storeu_ps(d + 0, s128);
		_mm_storeu_ps(d + 4, s128);
		d += 8;
		rem -= 4;
	}

	if (rem >= 2) {
		_mm_storeu_ps(d, s128);
		d += 4;
		rem -= 2;
	}

	if (rem) {
		d[0] = s[0];
		d[1] = s[1];
	}
}

void memset128(void* __restrict dst, uint32_t n128, const void* __restrict src)
{
	const __m128 s128 = _mm_loadu_ps((const float*)src);
	float* d = (float*)dst;

	uint32_t iter = n128 >> 2; // 4 128-bit values per iteration (== 16 floats / iter)
	while (iter-- > 0) {
		_mm_storeu_ps(d + 0, s128);
		_mm_storeu_ps(d + 4, s128);
		_mm_storeu_ps(d + 8, s128);
		_mm_storeu_ps(d + 12, s128);
		d += 16;
	}

	uint32_t rem = n128 & 3;
	if (rem >= 2) {
		_mm_storeu_ps(d + 0, s128);
		_mm_storeu_ps(d + 4, s128);
		d += 8;
		rem -= 2;
	}

	if (rem) {
		_mm_storeu_ps(d, s128);
	}
}

// Transforms 2 interleaved positions { x0, y0, x1, y1 } at once.
//   x' = (x * m[0] + m[4]) + y * m[2]
//   y' = (x * m[1] + m[5]) + y * m[3]
// NOTE: The evaluation order is part of the contract (see batchTransformPositions()). All code paths
// (SIMD body, SIMD tail and the non-SIMD fallback) use it so they produce bit-identical results.
static inline __m128 transformPositions2(__m128 xy, __m128 mtx0101, __m128 mtx2323, __m128 mtx4545)
{
	const __m128 xx = _mm_shuffle_ps(xy, xy, _MM_SHUFFLE(2, 2, 0, 0)); // { x0, x0, x1, x1 }
	const __m128 yy = _mm_shuffle_ps(xy, xy, _MM_SHUFFLE(3, 3, 1, 1)); // { y0, y0, y1, y1 }
	return _mm_add_ps(_mm_add_ps(_mm_mul_ps(xx, mtx0101), mtx4545), _mm_mul_ps(yy, mtx2323));
}

void batchTransformPositions(const float* __restrict src, uint32_t n, float* __restrict dst, const float* __restrict mtx)
{
	// Works directly on the interleaved { x, y } layout (no deinterleave/reinterleave shuffles).
	const __m128 mtx0123 = _mm_loadu_ps(mtx);
	const __m128 mtx45 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(mtx + 4));

	const __m128 mtx0101 = _mm_movelh_ps(mtx0123, mtx0123); // { m0, m1, m0, m1 }
	const __m128 mtx2323 = _mm_movehl_ps(mtx0123, mtx0123); // { m2, m3, m2, m3 }
	const __m128 mtx4545 = _mm_movelh_ps(mtx45, mtx45);     // { m4, m5, m4, m5 }

	const uint32_t iter = n >> 3;
	for (uint32_t i = 0; i < iter; ++i) {
		const __m128 xy01 = _mm_loadu_ps(src + 0);  // { x0, y0, x1, y1 }
		const __m128 xy23 = _mm_loadu_ps(src + 4);  // { x2, y2, x3, y3 }
		const __m128 xy45 = _mm_loadu_ps(src + 8);  // { x4, y4, x5, y5 }
		const __m128 xy67 = _mm_loadu_ps(src + 12); // { x6, y6, x7, y7 }

		_mm_storeu_ps(dst + 0, transformPositions2(xy01, mtx0101, mtx2323, mtx4545));
		_mm_storeu_ps(dst + 4, transformPositions2(xy23, mtx0101, mtx2323, mtx4545));
		_mm_storeu_ps(dst + 8, transformPositions2(xy45, mtx0101, mtx2323, mtx4545));
		_mm_storeu_ps(dst + 12, transformPositions2(xy67, mtx0101, mtx2323, mtx4545));

		src += 16;
		dst += 16;
	}

	uint32_t rem = n & 7;
	if (rem >= 4) {
		const __m128 xy01 = _mm_loadu_ps(src + 0);
		const __m128 xy23 = _mm_loadu_ps(src + 4);

		_mm_storeu_ps(dst + 0, transformPositions2(xy01, mtx0101, mtx2323, mtx4545));
		_mm_storeu_ps(dst + 4, transformPositions2(xy23, mtx0101, mtx2323, mtx4545));

		src += 8;
		dst += 8;
		rem -= 4;
	}

	if (rem >= 2) {
		const __m128 xy01 = _mm_loadu_ps(src);
		_mm_storeu_ps(dst, transformPositions2(xy01, mtx0101, mtx2323, mtx4545));

		src += 4;
		dst += 4;
		rem -= 2;
	}

	if (rem) {
		// Same evaluation order as transformPositions2()
		const float x = src[0];
		const float y = src[1];
		dst[0] = (x * mtx[0] + mtx[4]) + y * mtx[2];
		dst[1] = (x * mtx[1] + mtx[5]) + y * mtx[3];
	}
}
#else
void memset32(void* __restrict dst, uint32_t n, const void* __restrict src)
{
	const uint32_t s = *(const uint32_t*)src;
	uint32_t* d = (uint32_t*)dst;
	while (n-- > 0) {
		*d++ = s;
	}
}

void memset64(void* __restrict dst, uint32_t n64, const void* __restrict src)
{
	const uint32_t s0 = *((const uint32_t*)src + 0);
	const uint32_t s1 = *((const uint32_t*)src + 1);
	uint32_t* d = (uint32_t*)dst;
	while (n64-- > 0) {
		d[0] = s0;
		d[1] = s1;
		d += 2;
	}
}

void memset128(void* __restrict dst, uint32_t n128, const void* __restrict src)
{
	const uint32_t s0 = *((const uint32_t*)src + 0);
	const uint32_t s1 = *((const uint32_t*)src + 1);
	const uint32_t s2 = *((const uint32_t*)src + 2);
	const uint32_t s3 = *((const uint32_t*)src + 3);
	uint32_t* d = (uint32_t*)dst;
	while (n128-- > 0) {
		d[0] = s0;
		d[1] = s1;
		d[2] = s2;
		d[3] = s3;
		d += 4;
	}
}

void batchTransformPositions(const float* __restrict v, uint32_t n, float* __restrict p, const float* __restrict mtx)
{
	// NOTE: Same evaluation order as the SIMD version (x * m[0] + m[4]) + y * m[2] (which differs from
	// transformPos2D()) so both paths produce bit-identical results.
	for (uint32_t i = 0; i < n; ++i) {
		const uint32_t id = i << 1;
		const float x = v[id + 0];
		const float y = v[id + 1];
		p[id + 0] = (x * mtx[0] + mtx[4]) + y * mtx[2];
		p[id + 1] = (x * mtx[1] + mtx[5]) + y * mtx[3];
	}
}
#endif

void genQuadIndices_unaligned(uint16_t* dst, uint32_t n, uint16_t firstVertexID)
{
#if VG_CONFIG_ENABLE_SIMD && BX_CPU_X86
	BX_ALIGN_DECL(16, static const uint16_t delta[]) = {
		0, 1, 2, 0, 2, 3,
		4, 5, 6, 4, 6, 7,
		8, 9, 10, 8, 10, 11,
		12, 13, 14, 12, 14, 15
	};

	const __m128i xmm_delta0 = _mm_load_si128((const __m128i*)&delta[0]);
	const __m128i xmm_delta1 = _mm_load_si128((const __m128i*)&delta[8]);
	const __m128i xmm_delta2 = _mm_load_si128((const __m128i*)&delta[16]);

	const uint32_t numIter = n >> 2; // 4 quads per iteration
	for (uint32_t i = 0; i < numIter; ++i) {
		const __m128i id = _mm_set1_epi16(firstVertexID);

		const __m128i id0 = _mm_add_epi16(id, xmm_delta0);
		const __m128i id1 = _mm_add_epi16(id, xmm_delta1);
		const __m128i id2 = _mm_add_epi16(id, xmm_delta2);
		_mm_storeu_si128((__m128i*)(dst + 0), id0);
		_mm_storeu_si128((__m128i*)(dst + 8), id1);
		_mm_storeu_si128((__m128i*)(dst + 16), id2);

		dst += 24;
		firstVertexID += 16;
	}

	uint32_t rem = n & 3;
	switch (rem) {
	case 3:
		dst[0] = firstVertexID; dst[1] = firstVertexID + 1; dst[2] = firstVertexID + 2;
		dst[3] = firstVertexID; dst[4] = firstVertexID + 2; dst[5] = firstVertexID + 3;
		dst += 6;
		firstVertexID += 4;
	case 2:
		dst[0] = firstVertexID; dst[1] = firstVertexID + 1; dst[2] = firstVertexID + 2;
		dst[3] = firstVertexID; dst[4] = firstVertexID + 2; dst[5] = firstVertexID + 3;
		dst += 6;
		firstVertexID += 4;
	case 1:
		dst[0] = firstVertexID; dst[1] = firstVertexID + 1; dst[2] = firstVertexID + 2;
		dst[3] = firstVertexID; dst[4] = firstVertexID + 2; dst[5] = firstVertexID + 3;
		dst += 6;
		firstVertexID += 4;
	}
#else
	while (n-- > 0) {
		dst[0] = firstVertexID; dst[1] = firstVertexID + 1; dst[2] = firstVertexID + 2;
		dst[3] = firstVertexID; dst[4] = firstVertexID + 2; dst[5] = firstVertexID + 3;
		dst += 6;
		firstVertexID += 4;
	}
#endif
}

void batchTransformTextQuads(const float* __restrict quads, uint32_t n, const float* __restrict mtx, float* __restrict transformedVertices)
{
	// NOTE: quads and transformedVertices can be unaligned.
	// NOTE: Same evaluation order as transformPos2D() used by the scalar version below, i.e.
	//   x' = (m[0] * x + m[2] * y) + m[4]
	//   y' = (m[1] * x + m[3] * y) + m[5]
	// so both paths produce bit-identical results.
#if VG_CONFIG_ENABLE_SIMD && BX_CPU_X86
	const __m128 mtx0123 = _mm_loadu_ps(mtx);
	const __m128 mtx45 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(mtx + 4));

	const __m128 mtx0101 = _mm_movelh_ps(mtx0123, mtx0123); // { m0, m1, m0, m1 }
	const __m128 mtx2323 = _mm_movehl_ps(mtx0123, mtx0123); // { m2, m3, m2, m3 }
	const __m128 mtx4545 = _mm_movelh_ps(mtx45, mtx45);     // { m4, m5, m4, m5 }

	for (uint32_t i = 0; i < n; ++i) {
		const __m128 q = _mm_loadu_ps(quads); // { x0, y0, x1, y1 }

		// v0 = (x0, y0), v1 = (x1, y0), v2 = (x1, y1), v3 = (x0, y1)
		const __m128 x0011 = _mm_shuffle_ps(q, q, _MM_SHUFFLE(2, 2, 0, 0)); // { x0, x0, x1, x1 }
		const __m128 y0000 = _mm_shuffle_ps(q, q, _MM_SHUFFLE(1, 1, 1, 1)); // { y0, y0, y0, y0 }
		const __m128 x1100 = _mm_shuffle_ps(q, q, _MM_SHUFFLE(0, 0, 2, 2)); // { x1, x1, x0, x0 }
		const __m128 y1111 = _mm_shuffle_ps(q, q, _MM_SHUFFLE(3, 3, 3, 3)); // { y1, y1, y1, y1 }

		const __m128 v01 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(mtx0101, x0011), _mm_mul_ps(mtx2323, y0000)), mtx4545);
		const __m128 v23 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(mtx0101, x1100), _mm_mul_ps(mtx2323, y1111)), mtx4545);

		_mm_storeu_ps(transformedVertices + 0, v01);
		_mm_storeu_ps(transformedVertices + 4, v23);

		quads += 8;
		transformedVertices += 8;
	}
#else
	for (uint32_t i = 0; i < n; ++i) {
		const float* q = &quads[i * 8];
		const uint32_t s = i << 3;
		transformPos2D(q[0], q[1], mtx, &transformedVertices[s + 0]);
		transformPos2D(q[2], q[1], mtx, &transformedVertices[s + 2]);
		transformPos2D(q[2], q[3], mtx, &transformedVertices[s + 4]);
		transformPos2D(q[0], q[3], mtx, &transformedVertices[s + 6]);
	}
#endif
}

void batchTransformDrawIndices(const uint16_t* __restrict src, uint32_t n, uint16_t* __restrict dst, uint16_t delta)
{
	if (delta == 0) {
		bx::memCopy(dst, src, sizeof(uint16_t) * n);
		return;
	}

#if VG_CONFIG_ENABLE_SIMD && BX_CPU_X86
	const __m128i xmm_delta = _mm_set1_epi16(delta);

	const uint32_t iter32 = n >> 5;
	for (uint32_t i = 0; i < iter32; ++i) {
		const __m128i s0 = _mm_loadu_si128((const __m128i*)src);
		const __m128i s1 = _mm_loadu_si128((const __m128i*)(src + 8));
		const __m128i s2 = _mm_loadu_si128((const __m128i*)(src + 16));
		const __m128i s3 = _mm_loadu_si128((const __m128i*)(src + 24));

		const __m128i d0 = _mm_add_epi16(s0, xmm_delta);
		const __m128i d1 = _mm_add_epi16(s1, xmm_delta);
		const __m128i d2 = _mm_add_epi16(s2, xmm_delta);
		const __m128i d3 = _mm_add_epi16(s3, xmm_delta);

		// NOTE: Proper alignment of dst buffer isn't guaranteed because it's part of the global IndexBuffer.
		_mm_storeu_si128((__m128i*)dst, d0);
		_mm_storeu_si128((__m128i*)(dst + 8), d1);
		_mm_storeu_si128((__m128i*)(dst + 16), d2);
		_mm_storeu_si128((__m128i*)(dst + 24), d3);

		src += 32;
		dst += 32;
	}

	uint32_t rem = n & 31;
	if (rem >= 16) {
		const __m128i s0 = _mm_loadu_si128((const __m128i*)src);
		const __m128i s1 = _mm_loadu_si128((const __m128i*)(src + 8));

		const __m128i d0 = _mm_add_epi16(s0, xmm_delta);
		const __m128i d1 = _mm_add_epi16(s1, xmm_delta);

		_mm_storeu_si128((__m128i*)dst, d0);
		_mm_storeu_si128((__m128i*)(dst + 8), d1);

		src += 16;
		dst += 16;
		rem -= 16;
	}

	if (rem >= 8) {
		__m128i s0 = _mm_loadu_si128((const __m128i*)src);
		__m128i d0 = _mm_add_epi16(s0, xmm_delta);
		_mm_storeu_si128((__m128i*)dst, d0);

		src += 8;
		dst += 8;
		rem -= 8;
	}

	switch (rem) {
	case 7: *dst++ = *src++ + delta;
	case 6: *dst++ = *src++ + delta;
	case 5: *dst++ = *src++ + delta;
	case 4: *dst++ = *src++ + delta;
	case 3: *dst++ = *src++ + delta;
	case 2: *dst++ = *src++ + delta;
	case 1: *dst = *src + delta;
	}
#else
	for (uint32_t i = 0; i < n; ++i) {
		*dst++ = *src + delta;
		src++;
	}
#endif
}

void convertA8_to_RGBA8(uint32_t* rgba, const uint8_t* a8, uint32_t w, uint32_t h, uint32_t rgbColor)
{
	const uint32_t rgb0 = rgbColor & 0x00FFFFFF;

	int numPixels = w * h;
	for (int i = 0; i < numPixels; ++i) {
		*rgba++ = rgb0 | (((uint32_t)* a8) << 24);
		++a8;
	}
}
}
