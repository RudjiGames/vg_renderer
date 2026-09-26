#include <vg/stroker.h>
#include "vg_util.h"
#include "libtess2/tesselator.h"
#include <bx/allocator.h>
#include <bx/math.h>
#include <string.h> // memcpy

BX_PRAGMA_DIAGNOSTIC_IGNORED_MSVC(4127) // conditional expression is constant
BX_PRAGMA_DIAGNOSTIC_IGNORED_MSVC(4456) // declaration of X hides previous local decleration
BX_PRAGMA_DIAGNOSTIC_IGNORED_CLANG_GCC("-Wshadow")

#ifndef RSQRT_ALGORITHM
#	define RSQRT_ALGORITHM 1
#endif

#ifndef RCP_ALGORITHM
#	define RCP_ALGORITHM 1
#endif

namespace vg
{
struct Vec2
{
	float x, y;
};

inline Vec2 vec2Add(const Vec2& a, const Vec2& b)    { return{ a.x + b.x, a.y + b.y }; }
inline Vec2 vec2Sub(const Vec2& a, const Vec2& b)    { return{ a.x - b.x, a.y - b.y }; }
inline Vec2 vec2Scale(const Vec2& a, float s)        { return{ a.x * s, a.y * s }; }
inline Vec2 vec2PerpCCW(const Vec2& a)               { return{ -a.y, a.x }; }
inline Vec2 vec2PerpCW(const Vec2& a)                { return{ a.y, -a.x }; }
inline float vec2Cross(const Vec2& a, const Vec2& b) { return a.x * b.y - b.x * a.y; }
inline float vec2Dot(const Vec2& a, const Vec2& b)   { return a.x * b.x + a.y * b.y; }

// Direction from a to b
inline Vec2 vec2Dir(const Vec2& a, const Vec2& b)
{
	const float dx = b.x - a.x;
	const float dy = b.y - a.y;
	const float lenSqr = dx * dx + dy * dy;
	const float invLen = lenSqr < VG_EPSILON ? 0.0f : bx::rsqrt(lenSqr);
	return{ dx * invLen, dy * invLen };
}

// Unit length version of a (possibly zero) vector, used as the start/end direction of arcs.
// Zero vectors map to the +X axis, the same as bx::atan2(0, 0) == 0 did. Unlike vec2Dir() this
// uses an exact sqrt because arc points are expected to lie on the circle.
inline Vec2 vec2ArcDir(const Vec2& a)
{
	const float lenSqr = a.x * a.x + a.y * a.y;
	if (lenSqr == 0.0f) {
		return{ 1.0f, 0.0f };
	}

	const float invLen = 1.0f / bx::sqrt(lenSqr);
	return{ a.x * invLen, a.y * invLen };
}

// Rotate a by the angle whose cosine and sine are ca and sa.
inline Vec2 vec2Rotate(const Vec2& a, float ca, float sa)
{
	return{ ca * a.x - sa * a.y, sa * a.x + ca * a.y };
}

// Minimum |cross(d12, d01)| for which the miter extrusion vector (d01 - d12) / cross is used. Below
// that (almost collinear or almost reversed segments) the perpendicular of d01 is used instead. Since
// |d01 - d12| <= 2, this limits the extrusion vector's length to 2 / 0.01 = 200 times the extrusion
// distance, which vg.cpp relies on when culling paths against the scissor rect (kMaxExtrusionScale).
// NOTE: All extrusion vector calculations (scalar and SIMD) must use the same threshold.
static const float kMinExtrusionCross = 1.0f / 100.0f;

inline Vec2 calcExtrusionVector(const Vec2& d01, const Vec2& d12)
{
	// v is the vector from the path point to the outline point, assuming a stroke width of 1.0.
	// Equation obtained by solving the intersection of the 2 line segments. d01 and d12 are 
	// assumed to be normalized.
	Vec2 v = vec2PerpCCW(d01);
	const float cross = vec2Cross(d12, d01);
	if (bx::abs(cross) > kMinExtrusionCross) {
		v = vec2Scale(vec2Sub(d01, d12), (1.0f / cross));
	}

	return v;
}

#if VG_CONFIG_ENABLE_SIMD && BX_CPU_X86
static const __m128 vec2_perpCCW_xorMask = _mm_castsi128_ps(_mm_set_epi32(0, 0, 0, 0x80000000));

static inline __m128 xmm_vec2_rotCCW90(const __m128 a)
{
	__m128 ayx = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 2, 0, 1)); // { a.y, a.x, DC, DC }
	return _mm_xor_ps(ayx, vec2_perpCCW_xorMask); // { -a.y, a.x, DC, DC }
}

static inline float xmm_vec2_cross(const __m128 a, const __m128 b)
{
	const __m128 axy_bxy = _mm_movelh_ps(a, b); // { a.x, a.y, b.x, b.y }
	const __m128 byx_ayx = _mm_shuffle_ps(axy_bxy, axy_bxy, _MM_SHUFFLE(0, 1, 2, 3)); // { b.y, b.x, a.y, a.x }
	const __m128 axby_aybx = _mm_mul_ps(axy_bxy, byx_ayx); // { a.x * b.y, a.y * b.x, b.x * a.y, b.y * a.x }
	const __m128 bxay = _mm_shuffle_ps(axby_aybx, axby_aybx, _MM_SHUFFLE(1, 1, 1, 1)); // { a.y * b.x, a.y * b.x, a.y * b.x, a.y * b.x }
	const __m128 cross = _mm_sub_ss(axby_aybx, bxay);
	return _mm_cvtss_f32(cross);
}

static inline __m128 xmm_vec2_dir(const __m128 a, const __m128 b)
{
	const __m128 dxy = _mm_sub_ps(b, a); // { dx, dy, DC, DC }
	const __m128 dxySqr = _mm_mul_ps(dxy, dxy); // { dx * dx, dy * dy, DC, DC }
	const __m128 dySqr = _mm_shuffle_ps(dxySqr, dxySqr, _MM_SHUFFLE(1, 1, 1, 1)); // { dy * dy, dy * dy, dy * dy, dy * dy }
	const float lenSqr = _mm_cvtss_f32(_mm_add_ss(dxySqr, dySqr));
	__m128 dir = _mm_setzero_ps();
	if (lenSqr >= VG_EPSILON) {
		const __m128 invLen = _mm_set_ps1(bx::rsqrt(lenSqr));
		dir = _mm_mul_ps(dxy, invLen);
	}
	return dir;
}

static inline __m128 xmm_calcExtrusionVector(const __m128 d01, const __m128 d12)
{
	const float cross = xmm_vec2_cross(d12, d01);
	return (bx::abs(cross) > kMinExtrusionCross) ? _mm_mul_ps(_mm_sub_ps(d01, d12), _mm_set_ps1(1.0f / cross)) : xmm_vec2_rotCCW90(d01);
}

// Alternative implementations of the vector reciprocal (square root). 0 = exact (division),
// 1 = hardware estimate (~12 bits), 2 = hardware estimate refined with one Newton-Raphson step (~22 bits).
static inline __m128 xmm_rsqrt(__m128 a)
{
#if RSQRT_ALGORITHM == 0
	const __m128 xmm_one = _mm_set1_ps(1.0f);
	const __m128 res = _mm_div_ps(xmm_one, _mm_sqrt_ps(a));
#elif RSQRT_ALGORITHM == 1
	const __m128 res = _mm_rsqrt_ps(a);
#elif RSQRT_ALGORITHM == 2
	// Newton/Raphson: x1 = 0.5 * x0 * (3 - a * x0 * x0)
	const __m128 xmm_half = _mm_set1_ps(0.5f);
	const __m128 xmm_three = _mm_set1_ps(3.0f);
	const __m128 rsqrtEst = _mm_rsqrt_ps(a);
	const __m128 iter0 = _mm_mul_ps(a, rsqrtEst);
	const __m128 iter1 = _mm_mul_ps(iter0, rsqrtEst);
	const __m128 half_rsqrt = _mm_mul_ps(xmm_half, rsqrtEst);
	const __m128 three_sub_iter1 = _mm_sub_ps(xmm_three, iter1);
	const __m128 res = _mm_mul_ps(half_rsqrt, three_sub_iter1);
#else
#	error "Unknown RSQRT_ALGORITHM"
#endif

	return res;
}

static inline __m128 xmm_rcp(__m128 a)
{
#if RCP_ALGORITHM == 0
	const __m128 xmm_one = _mm_set1_ps(1.0f);
	const __m128 inv_a = _mm_div_ps(xmm_one, a);
#elif RCP_ALGORITHM == 1
	const __m128 inv_a = _mm_rcp_ps(a);
#elif RCP_ALGORITHM == 2
	// Newton/Raphson: x1 = x0 * (2 - a * x0)
	const __m128 xmm_two = _mm_set1_ps(2.0f);
	const __m128 rcpEst = _mm_rcp_ps(a);
	const __m128 inv_a = _mm_mul_ps(rcpEst, _mm_sub_ps(xmm_two, _mm_mul_ps(a, rcpEst)));
#else
#	error "Unknown RCP_ALGORITHM"
#endif

	return inv_a;
}
#endif

// Every libtess2 allocation is prefixed by a header. Blocks are carved out of the scratch buffer while it has
// room, and fall back to the heap once it's full (heap blocks are tracked so they can be freed on reset).
// NOTE: Heap allocations are bounded by VG_CONFIG_LIBTESS2_HEAP_LIMIT because on some (self-intersecting)
// inputs libtess2's sweep never terminates and keeps allocating. The limit makes such tessellations fail
// instead of eating all available memory.
struct libtess2AllocHeader
{
	libtess2AllocHeader* m_Next;
	libtess2AllocHeader* m_Prev;
	uint32_t m_Size;
	uint32_t m_Heap;
};

static const uint32_t kLibtess2HeaderSize = 32;
static_assert(sizeof(libtess2AllocHeader) <= kLibtess2HeaderSize, "libtess2 allocation header doesn't fit.");

struct libtess2Allocator
{
	bx::AllocatorI* m_Allocator;
	uint8_t* m_Buffer;
	uint32_t m_Capacity;
	uint32_t m_Size;
	libtess2AllocHeader* m_HeapList;
	uint64_t m_HeapSize; // Total size of the live heap blocks
};

// Frees all heap blocks (libtess2 doesn't free everything when tessellation fails) and rewinds the scratch buffer.
static void libtess2Reset(libtess2Allocator* alloc)
{
	while (alloc->m_HeapList) {
		libtess2AllocHeader* header = alloc->m_HeapList;
		alloc->m_HeapList = header->m_Next;
		bx::alignedFree(alloc->m_Allocator, header, 16);
	}

	alloc->m_Size = 0;
	alloc->m_HeapSize = 0;
}

#if VG_CONFIG_LIBTESS2_SCRATCH_BUFFER
static inline libtess2AllocHeader* libtess2GetHeader(void* ptr)
{
	return (libtess2AllocHeader*)((uint8_t*)ptr - kLibtess2HeaderSize);
}

static void* libtess2Alloc(void* userData, uint32_t size)
{
	libtess2Allocator* alloc = (libtess2Allocator*)userData;

	// Align all allocations to 16 bytes
	const uint32_t offset = (alloc->m_Size & ~0x0F) + ((alloc->m_Size & 0x0F) != 0 ? 0x10 : 0);

	libtess2AllocHeader* header;
	if ((uint64_t)offset + kLibtess2HeaderSize + size <= alloc->m_Capacity) {
		header = (libtess2AllocHeader*)&alloc->m_Buffer[offset];
		header->m_Next = nullptr;
		header->m_Prev = nullptr;
		header->m_Heap = 0;
		alloc->m_Size = offset + kLibtess2HeaderSize + size;
	} else {
		if (alloc->m_HeapSize + size > VG_CONFIG_LIBTESS2_HEAP_LIMIT || (uint64_t)kLibtess2HeaderSize + size > UINT32_MAX) {
			return nullptr;
		}

		header = (libtess2AllocHeader*)bx::alignedAlloc(alloc->m_Allocator, kLibtess2HeaderSize + size, 16);
		if (!header) {
			return nullptr;
		}

		header->m_Next = alloc->m_HeapList;
		header->m_Prev = nullptr;
		header->m_Heap = 1;
		if (alloc->m_HeapList) {
			alloc->m_HeapList->m_Prev = header;
		}
		alloc->m_HeapList = header;
		alloc->m_HeapSize += size;
	}

	header->m_Size = size;
	return (uint8_t*)header + kLibtess2HeaderSize;
}

static void libtess2Free(void* userData, void* ptr)
{
	if (!ptr) {
		return;
	}

	// Scratch blocks are released all at once by libtess2Reset().
	libtess2AllocHeader* header = libtess2GetHeader(ptr);
	if (!header->m_Heap) {
		return;
	}

	libtess2Allocator* alloc = (libtess2Allocator*)userData;
	if (header->m_Prev) {
		header->m_Prev->m_Next = header->m_Next;
	} else {
		alloc->m_HeapList = header->m_Next;
	}
	if (header->m_Next) {
		header->m_Next->m_Prev = header->m_Prev;
	}

	alloc->m_HeapSize -= header->m_Size;
	bx::alignedFree(alloc->m_Allocator, header, 16);
}

static void* libtess2Realloc(void* userData, void* ptr, uint32_t size)
{
	if (!ptr) {
		return libtess2Alloc(userData, size);
	}

	libtess2Allocator* alloc = (libtess2Allocator*)userData;
	libtess2AllocHeader* header = libtess2GetHeader(ptr);

	// The last scratch allocation can be resized in place.
	if (!header->m_Heap) {
		const uint32_t offset = (uint32_t)((uint8_t*)ptr - alloc->m_Buffer);
		if (offset + header->m_Size == alloc->m_Size
		&&  (uint64_t)offset + size <= alloc->m_Capacity) {
			header->m_Size = size;
			alloc->m_Size = offset + size;
			return ptr;
		}
	}

	void* mem = libtess2Alloc(userData, size);
	if (!mem) {
		return nullptr;
	}

	bx::memCopy(mem, ptr, bx::min<uint32_t>(header->m_Size, size));
	libtess2Free(userData, ptr);
	return mem;
}
#endif // VG_CONFIG_LIBTESS2_SCRATCH_BUFFER

// Arc of a round join (see calcRoundJoinArcs()).
struct RoundJoinArc
{
	Vec2 m_ArcDir;           // Direction of the first arc point
	float m_CosDa;           // cos/sin of the angle between successive arc points
	float m_SinDa;
	uint32_t m_NumArcPoints;
};

// Max number of vertices of a concave polygon to triangulate by ear clipping instead of libtess2
// (see triangulateSimplePolygon()).
static const uint32_t kMaxSimplePolygonVertices = 256;

struct Stroker
{
	bx::AllocatorI* m_Allocator;
	Vec2* m_PosBuffer;
	uint32_t* m_ColorBuffer;
	uint16_t* m_IndexBuffer;
	uint16_t* m_FanIndexBuffer;   // Triangle fan indices (0, i, i + 1) used by strokerConvexFill(). Grow-only.
	uint32_t m_FanTriCapacity;    // Number of triangles in m_FanIndexBuffer
	Vec2* m_SegmentBuffer;        // Per segment directions and per vertex extrusion vectors used by the polyline strokers. Grow-only.
	uint32_t m_SegmentCapacity;   // Number of Vec2 in m_SegmentBuffer
	RoundJoinArc* m_JoinBuffer;   // Per join arc parameters of round joins. Grow-only.
	uint32_t m_JoinCapacity;      // Number of elements in m_JoinBuffer
	uint32_t* m_VertexMap;        // Tesselator vertex -> output vertex map used by strokerConcaveFillEndAA(). Grow-only.
	uint32_t m_VertexMapCapacity;
	uint32_t* m_EdgeHash;         // Keys and values of the edge hash table used by fixFlippedTriangles(). Grow-only.
	uint32_t m_EdgeHashCapacity;  // Number of slots in m_EdgeHash
	uint32_t* m_ClampScratch;     // Scratch memory used by clampInset(). Grow-only.
	uint32_t m_ClampScratchCapacity;
	Vec2 m_SimplePolyVertices[kMaxSimplePolygonVertices];                  // See triangulateSimplePolygon()
	uint16_t m_SimplePolyTriangles[(kMaxSimplePolygonVertices - 2) * 3];
	uint16_t m_SimplePolyBoundary[kMaxSimplePolygonVertices + 2];           // Boundary vertex IDs + contour (first, count)
	Vec2* m_ContourVertices;      // Copy of the contours added with strokerConcaveFillAddContour()
	uint32_t m_NumContourVertices;
	uint32_t m_ContourVertexCapacity;
	uint32_t* m_ContourSizes;
	uint32_t m_NumContours;
	uint32_t m_ContourCapacity;
	uint32_t m_NumVertices;
	uint32_t m_NumIndices;
	uint32_t m_VertexCapacity;
	uint32_t m_IndexCapacity;
	TESStesselator* m_Tesselator;
	libtess2Allocator m_libTessAllocator;
	float m_FringeWidth;
	float m_Scale;
	float m_TesselationTolerance;
};

static void resetGeometry(Stroker* stroker);
static void expandIB(Stroker* stroker, uint32_t n);
static void expandVB(Stroker* stroker, uint32_t n);

template<bool _Closed, LineCap::Enum _LineCap, LineJoin::Enum _LineJoin>
static void polylineStroke(Stroker* stroker, Mesh* mesh, const Vec2* vtx, uint32_t numPathVertices, float strokeWidth, const StrokerSink* sink);
template<bool _Closed, LineCap::Enum _LineCap, LineJoin::Enum _LineJoin>
static void polylineStrokeAA(Stroker* stroker, Mesh* mesh, const Vec2* vtx, uint32_t numPathVertices, float strokeWidth, Color color, const StrokerSink* sink);
template<LineCap::Enum _LineCap, LineJoin::Enum _LineJoin>
static void polylineStrokeAAThin(Stroker* stroker, Mesh* mesh, const Vec2* vtx, uint32_t numPathVertices, Color color, bool closed, const StrokerSink* sink);

// Helpers for writing geometry through local pointers. The caller is responsible for reserving
// enough space up front (see beginGeometry()).
template<uint32_t N>
static BX_FORCE_INLINE Vec2* copyPos(Vec2* dst, const Vec2* src)
{
	memcpy(dst, src, sizeof(Vec2) * N);
	return dst + N;
}

template<uint32_t N>
static BX_FORCE_INLINE uint32_t* copyColor(uint32_t* dst, const uint32_t* src)
{
	memcpy(dst, src, sizeof(uint32_t) * N);
	return dst + N;
}

template<uint32_t N>
static BX_FORCE_INLINE uint16_t* copyIndices(uint16_t* dst, const uint16_t* src)
{
	memcpy(dst, src, sizeof(uint16_t) * N);
	return dst + N;
}

// Destination of the geometry generated by a stroker function. Either the stroker's internal buffers or
// the buffers returned by the caller's StrokerSink.
struct GeometryOutput
{
	Vec2* m_Pos;
	uint32_t* m_Color;
	uint16_t* m_Index;
	uint32_t m_NumVertices; // Exact amount of geometry which will be generated
	uint32_t m_NumIndices;
	uint16_t m_BaseVertex;
	bool m_External;
};

// Allocates space for exactly numVertices vertices and numIndices indices. The geometry is written through local
// pointers starting at out->m_Pos/m_Color/m_Index, with indices relative to out->m_Pos. endGeometry() finishes it.
static void beginGeometry(Stroker* stroker, const StrokerSink* sink, uint32_t numVertices, uint32_t numIndices, bool generatesColors, GeometryOutput* out)
{
	out->m_NumVertices = numVertices;
	out->m_NumIndices = numIndices;

	if (sink) {
		StrokerOutput so;
		if (sink->m_AllocFn(sink->m_UserData, numVertices, numIndices, &so)) {
			VG_CHECK(!generatesColors || so.m_ColorBuffer, "A color buffer is required");
			BX_UNUSED(generatesColors);
			out->m_Pos = (Vec2*)so.m_PosBuffer;
			out->m_Color = so.m_ColorBuffer;
			out->m_Index = so.m_IndexBuffer;
			out->m_BaseVertex = so.m_BaseVertex;
			out->m_External = true;
			return;
		}
	}

	resetGeometry(stroker);
	expandVB(stroker, numVertices);
	expandIB(stroker, numIndices);
	out->m_Pos = stroker->m_PosBuffer;
	out->m_Color = stroker->m_ColorBuffer;
	out->m_Index = stroker->m_IndexBuffer;
	out->m_BaseVertex = 0;
	out->m_External = false;
}

// Adds base to n indices (in place).
static void addIndexBase(uint16_t* idx, uint32_t n, uint16_t base)
{
	if (base == 0) {
		return;
	}

	uint32_t i = 0;
#if VG_CONFIG_ENABLE_SIMD && BX_CPU_X86
	const __m128i xmm_base = _mm_set1_epi16((short)base);
	for (; i + 8 <= n; i += 8) {
		const __m128i v = _mm_loadu_si128((const __m128i*)&idx[i]);
		_mm_storeu_si128((__m128i*)&idx[i], _mm_add_epi16(v, xmm_base));
	}
#endif
	for (; i < n; ++i) {
		idx[i] = (uint16_t)(idx[i] + base);
	}
}

// Finishes the geometry started with beginGeometry(). dstPos/dstIndex point past the last written vertex/index.
static void endGeometry(Stroker* stroker, const GeometryOutput* out, const Vec2* dstPos, const uint16_t* dstIndex, bool hasColors, Mesh* mesh)
{
	const uint32_t numVertices = (uint32_t)(dstPos - out->m_Pos);
	const uint32_t numIndices = (uint32_t)(dstIndex - out->m_Index);
	VG_CHECK(numVertices == out->m_NumVertices && numIndices == out->m_NumIndices, "Generated geometry (%u vertices, %u indices) doesn't match the reserved space (%u vertices, %u indices)", numVertices, numIndices, out->m_NumVertices, out->m_NumIndices);

	if (out->m_External) {
		addIndexBase(out->m_Index, numIndices, out->m_BaseVertex);
	} else {
		stroker->m_NumVertices = numVertices;
		stroker->m_NumIndices = numIndices;
	}

	mesh->m_PosBuffer = &out->m_Pos[0].x;
	mesh->m_ColorBuffer = hasColors ? out->m_Color : nullptr;
	mesh->m_IndexBuffer = out->m_Index;
	mesh->m_NumVertices = numVertices;
	mesh->m_NumIndices = numIndices;
}

// Inner bevels: the inner corner of a join is the intersection of the inner edges of its 2 segments, which is
// projDist = |dot(d12, v_s)| behind the join along both segments (v_s is the extrusion vector scaled by the half
// stroke width). If one of the segments is shorter than that (sharp turns of thick strokes), the intersection is far
// away from the stroke and the join geometry creates a long spike. Such joins use 2 inner vertices instead (the inner
// corners of the 2 segments, like NanoVG's inner bevels), connected through the join point. The inner sides of the 2
// segments overlap in that case.
static BX_FORCE_INLINE bool needsInnerBevel(const Vec2* vtx, uint32_t numVertices, uint32_t i, float projDist)
{
	const Vec2 d01 = vec2Sub(vtx[i], vtx[i != 0 ? i - 1 : numVertices - 1]);
	const Vec2 d12 = vec2Sub(vtx[i + 1 < numVertices ? i + 1 : 0], vtx[i]);
	return projDist * projDist > bx::min(d01.x * d01.x + d01.y * d01.y, d12.x * d12.x + d12.y * d12.y);
}

// Calculates the direction of each segment of the polyline: dirs[i] = vec2Dir(vtx[i], vtx[i + 1]) for
// i in [0, numVertices - 1) and, for closed paths, the closing segment dirs[numVertices - 1] = vec2Dir(vtx[numVertices - 1], vtx[0]).
// Also calculates the extrusion vector of each join: ext[i] = calcExtrusionVector(dirs[i - 1], dirs[i]) for
// i in [1, numVertices - 1) and, for closed paths, ext[numVertices - 1] and ext[0] (using the closing segment).
// The results are bit-identical to calling vec2Dir()/calcExtrusionVector() for each segment/join (the SIMD
// version performs exactly the same IEEE operations, 4 segments at a time; bx::rsqrt() is 1.0f / sqrt() on SSE).
// Also decides which joins need an inner bevel (innerBevel[i] != 0, see needsInnerBevel()) for the given half stroke
// width (the one the join loops scale the extrusion vectors with to find the inner corner) and returns their number.
// The flags are calculated once and used both for the geometry budget and by the join loops, so they always match.
// All arrays point into the stroker's scratch buffer (valid until the next call).
static uint32_t calcSegmentDirsAndExtrusions(Stroker* stroker, const Vec2* vtx, uint32_t numVertices, bool closed, float innerBevelHsw, const Vec2** dirsOut, const Vec2** extOut, const uint8_t** innerBevelOut)
{
	VG_CHECK(numVertices >= 2, "Invalid number of vertices");
	const uint32_t numDirs = closed ? numVertices : numVertices - 1;

	// Layout: [dirs[0 .. numDirs) + 4 padding] [ext[0 .. numDirs) + 4 padding] [innerBevel[0 .. numDirs) + 4 padding]
	// The padding allows the SIMD code to always write 4 elements at a time.
	const uint32_t required = numDirs * 2 + 8 + (numDirs + 4 + 7) / 8;
	if (required > stroker->m_SegmentCapacity) {
		const uint32_t newCapacity = bx::max<uint32_t>(required, stroker->m_SegmentCapacity + (stroker->m_SegmentCapacity >> 1));
		// The old contents aren't needed so free the old buffer first.
		if (stroker->m_SegmentBuffer) {
			bx::alignedFree(stroker->m_Allocator, stroker->m_SegmentBuffer, 16);
		}
		stroker->m_SegmentBuffer = (Vec2*)bx::alignedAlloc(stroker->m_Allocator, sizeof(Vec2) * newCapacity, 16);
		stroker->m_SegmentCapacity = newCapacity;
	}

	Vec2* dirs = stroker->m_SegmentBuffer;
	Vec2* ext = dirs + numDirs + 4;
	uint8_t* innerBevel = (uint8_t*)(ext + numDirs + 4);
	uint32_t numInnerBevelJoins = 0;

#if VG_CONFIG_ENABLE_SIMD && BX_CPU_X86
	const __m128 xmm_one = _mm_set1_ps(1.0f);
	const __m128 xmm_epsilon = _mm_set1_ps(VG_EPSILON);
	const __m128 xmm_minCross = _mm_set1_ps(kMinExtrusionCross);
	const __m128 xmm_absMask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
	const __m128 xmm_signMask = _mm_castsi128_ps(_mm_set1_epi32((int)0x80000000));

	// The direction of the previous segment is in the last lane of prevDirX/Y. For the first join of a
	// closed path it's the closing segment.
	const Vec2 closingDir = closed ? vec2Dir(vtx[numVertices - 1], vtx[0]) : Vec2{ 0.0f, 0.0f };
	__m128 prevDirX = _mm_set1_ps(closingDir.x);
	__m128 prevDirY = _mm_set1_ps(closingDir.y);

	// Squared length of the previous segment (for the inner bevel test). There's no join at the first vertex of an
	// open path (its flag is cleared below).
	const Vec2 closingSeg = vec2Sub(vtx[0], vtx[numVertices - 1]);
	__m128 prevLenSqr = _mm_set1_ps(closingSeg.x * closingSeg.x + closingSeg.y * closingSeg.y);
	const __m128 xmm_hsw = _mm_set1_ps(innerBevelHsw);

	for (uint32_t i = 0; i < numDirs; i += 4) {
		// Load the start (a) and end (b) points of the 4 segments (SoA).
		__m128 ax, ay, bx_, by;
		if (i + 4 < numVertices) {
			const float* src = &vtx[i].x;
			const __m128 a01 = _mm_loadu_ps(src);     // { p0.x, p0.y, p1.x, p1.y }
			const __m128 a23 = _mm_loadu_ps(src + 4); // { p2.x, p2.y, p3.x, p3.y }
			const __m128 b01 = _mm_loadu_ps(src + 2); // { p1.x, p1.y, p2.x, p2.y }
			const __m128 b23 = _mm_loadu_ps(src + 6); // { p3.x, p3.y, p4.x, p4.y }
			ax = _mm_shuffle_ps(a01, a23, _MM_SHUFFLE(2, 0, 2, 0));
			ay = _mm_shuffle_ps(a01, a23, _MM_SHUFFLE(3, 1, 3, 1));
			bx_ = _mm_shuffle_ps(b01, b23, _MM_SHUFFLE(2, 0, 2, 0));
			by = _mm_shuffle_ps(b01, b23, _MM_SHUFFLE(3, 1, 3, 1));
		} else {
			// Last group. It might include the closing segment. Points past the end are replaced by
			// the first point (the lanes of non-existing segments are ignored).
			uint32_t id[5];
			for (uint32_t k = 0; k < 5; ++k) {
				id[k] = i + k < numVertices ? i + k : 0;
			}
			ax = _mm_setr_ps(vtx[id[0]].x, vtx[id[1]].x, vtx[id[2]].x, vtx[id[3]].x);
			ay = _mm_setr_ps(vtx[id[0]].y, vtx[id[1]].y, vtx[id[2]].y, vtx[id[3]].y);
			bx_ = _mm_setr_ps(vtx[id[1]].x, vtx[id[2]].x, vtx[id[3]].x, vtx[id[4]].x);
			by = _mm_setr_ps(vtx[id[1]].y, vtx[id[2]].y, vtx[id[3]].y, vtx[id[4]].y);
		}

		// Segment directions (vec2Dir())
		const __m128 dx = _mm_sub_ps(bx_, ax);
		const __m128 dy = _mm_sub_ps(by, ay);
		const __m128 lenSqr = _mm_add_ps(_mm_mul_ps(dx, dx), _mm_mul_ps(dy, dy));
		const __m128 invLen = _mm_andnot_ps(_mm_cmplt_ps(lenSqr, xmm_epsilon), _mm_div_ps(xmm_one, _mm_sqrt_ps(lenSqr)));
		const __m128 d12x = _mm_mul_ps(dx, invLen);
		const __m128 d12y = _mm_mul_ps(dy, invLen);

		// Directions of the previous segments: { prev[3], cur[0], cur[1], cur[2] }
		const __m128 tx = _mm_shuffle_ps(prevDirX, d12x, _MM_SHUFFLE(0, 0, 3, 3));
		const __m128 ty = _mm_shuffle_ps(prevDirY, d12y, _MM_SHUFFLE(0, 0, 3, 3));
		const __m128 d01x = _mm_shuffle_ps(tx, d12x, _MM_SHUFFLE(2, 1, 2, 0));
		const __m128 d01y = _mm_shuffle_ps(ty, d12y, _MM_SHUFFLE(2, 1, 2, 0));

		// Extrusion vectors (calcExtrusionVector()). cross = vec2Cross(d12, d01)
		const __m128 cross = _mm_sub_ps(_mm_mul_ps(d12x, d01y), _mm_mul_ps(d01x, d12y));
		const __m128 useMiter = _mm_cmpgt_ps(_mm_and_ps(cross, xmm_absMask), xmm_minCross);
		const __m128 invCross = _mm_div_ps(xmm_one, cross);
		const __m128 miterX = _mm_mul_ps(_mm_sub_ps(d01x, d12x), invCross);
		const __m128 miterY = _mm_mul_ps(_mm_sub_ps(d01y, d12y), invCross);

		// Fallback: vec2PerpCCW(d01) = { -d01.y, d01.x }
		const __m128 vx = _mm_or_ps(_mm_and_ps(useMiter, miterX), _mm_andnot_ps(useMiter, _mm_xor_ps(d01y, xmm_signMask)));
		const __m128 vy = _mm_or_ps(_mm_and_ps(useMiter, miterY), _mm_andnot_ps(useMiter, d01x));

		float* dstDir = &dirs[i].x;
		_mm_storeu_ps(dstDir, _mm_unpacklo_ps(d12x, d12y));
		_mm_storeu_ps(dstDir + 4, _mm_unpackhi_ps(d12x, d12y));

		float* dstExt = &ext[i].x;
		_mm_storeu_ps(dstExt, _mm_unpacklo_ps(vx, vy));
		_mm_storeu_ps(dstExt + 4, _mm_unpackhi_ps(vx, vy));

		// Inner bevels (needsInnerBevel()): projDist = dot(d12, v * hsw), limit = min(lenSqr[i - 1], lenSqr[i])
		const __m128 projDist = _mm_add_ps(_mm_mul_ps(d12x, _mm_mul_ps(vx, xmm_hsw)), _mm_mul_ps(d12y, _mm_mul_ps(vy, xmm_hsw)));
		const __m128 tl = _mm_shuffle_ps(prevLenSqr, lenSqr, _MM_SHUFFLE(0, 0, 3, 3));
		const __m128 lenSqr01 = _mm_shuffle_ps(tl, lenSqr, _MM_SHUFFLE(2, 1, 2, 0));
		uint32_t bevelMask = (uint32_t)_mm_movemask_ps(_mm_cmpgt_ps(_mm_mul_ps(projDist, projDist), _mm_min_ps(lenSqr01, lenSqr)));
		if (i == 0 && !closed) {
			bevelMask &= ~1u; // No join at the first vertex of an open path
		}
		if (i + 4 > numDirs) {
			bevelMask &= (1u << (numDirs - i)) - 1; // Past the last segment
		}
		static const uint32_t kMaskToBytes[16] = {
			0x00000000, 0x00000001, 0x00000100, 0x00000101, 0x00010000, 0x00010001, 0x00010100, 0x00010101,
			0x01000000, 0x01000001, 0x01000100, 0x01000101, 0x01010000, 0x01010001, 0x01010100, 0x01010101
		};
		static const uint8_t kMaskBits[16] = { 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4 };
		memcpy(&innerBevel[i], &kMaskToBytes[bevelMask], 4); // little endian (x86): byte k = lane k
		numInnerBevelJoins += kMaskBits[bevelMask];

		prevDirX = d12x;
		prevDirY = d12y;
		prevLenSqr = lenSqr;
	}
#else
	for (uint32_t i = 0; i < numDirs; ++i) {
		dirs[i] = vec2Dir(vtx[i], vtx[i + 1 < numVertices ? i + 1 : 0]);
	}

	if (closed) {
		ext[0] = calcExtrusionVector(dirs[numDirs - 1], dirs[0]);
	}

	for (uint32_t i = 1; i < numDirs; ++i) {
		ext[i] = calcExtrusionVector(dirs[i - 1], dirs[i]);
	}

	innerBevel[0] = 0; // No join at the first vertex of an open path
	for (uint32_t i = closed ? 0 : 1; i < numDirs; ++i) {
		const Vec2 v_s = vec2Scale(ext[i], innerBevelHsw);
		const float projDist = dirs[i].x * v_s.x + dirs[i].y * v_s.y;
		innerBevel[i] = needsInnerBevel(vtx, numVertices, i, projDist) ? 1 : 0;
		numInnerBevelJoins += innerBevel[i];
	}
#endif

	*dirsOut = dirs;
	*extOut = ext;
	*innerBevelOut = innerBevel;
	return numInnerBevelJoins;
}

// Calculates the arc of each round join in [firstJoin, lastJoin) (the calculations the join loops used to perform
// for each join) so that the exact amount of geometry is known before generating it. projScale is the scale
// of the extrusion vector used to determine the inner corner (hsw or hsw_aa, see the join loops). Returns the
// arcs (indexed by join/segment ID) and the total number of arc points.
// Returns how many of the inner bevel joins (see calcSegmentDirsAndExtrusions()) are bevel/round joins (miter joins
// and round joins generated as miters use the miter geometry).
static uint32_t countInnerBevelFanJoins(const uint8_t* innerBevel, uint32_t numInnerBevelJoins, uint32_t firstJoin, uint32_t lastJoin, LineJoin::Enum lineJoin, const RoundJoinArc* arcs)
{
	if (numInnerBevelJoins == 0 || lineJoin == LineJoin::Miter) {
		return 0;
	}

	if (lineJoin == LineJoin::Bevel) {
		return numInnerBevelJoins;
	}

	uint32_t numFan = 0;
	for (uint32_t i = firstJoin; i < lastJoin; ++i) {
		numFan += (innerBevel[i] && arcs[i].m_NumArcPoints != 0) ? 1 : 0;
	}
	return numFan;
}

// Indices connecting the end of the previous segment (IDs p*) to the start of a join (IDs t*).
static BX_FORCE_INLINE uint16_t* connectJoin2(uint16_t* dst, uint16_t pL, uint16_t pR, uint16_t tL, uint16_t tR)
{
	const uint16_t id[6] = { pL, pR, tR, pL, tR, tL };
	return copyIndices<6>(dst, id);
}

static BX_FORCE_INLINE uint16_t* connectJoin3(uint16_t* dst, uint16_t pLA, uint16_t pM, uint16_t pRA, uint16_t tLA, uint16_t tM, uint16_t tRA)
{
	const uint16_t id[12] = { pLA, pM, tM, pLA, tM, tLA, pM, pRA, tRA, pM, tRA, tM };
	return copyIndices<12>(dst, id);
}

static BX_FORCE_INLINE uint16_t* connectJoin4(uint16_t* dst, uint16_t pLA, uint16_t pL, uint16_t pR, uint16_t pRA, uint16_t tLA, uint16_t tL, uint16_t tR, uint16_t tRA)
{
	const uint16_t id[18] = { pLA, pL, tL, pLA, tL, tLA, pL, pR, tR, pL, tR, tL, pR, pRA, tRA, pR, tRA, tR };
	return copyIndices<18>(dst, id);
}

static const RoundJoinArc* calcRoundJoinArcs(Stroker* stroker, const Vec2* dirs, const Vec2* ext, uint32_t numDirs, uint32_t firstJoin, uint32_t lastJoin, float projScale, float da, uint32_t* totalArcPoints)
{
	if (lastJoin > stroker->m_JoinCapacity) {
		const uint32_t newCapacity = bx::max<uint32_t>(lastJoin, stroker->m_JoinCapacity + (stroker->m_JoinCapacity >> 1));
		if (stroker->m_JoinBuffer) {
			bx::alignedFree(stroker->m_Allocator, stroker->m_JoinBuffer, 16);
		}
		stroker->m_JoinBuffer = (RoundJoinArc*)bx::alignedAlloc(stroker->m_Allocator, sizeof(RoundJoinArc) * newCapacity, 16);
		stroker->m_JoinCapacity = newCapacity;
	}

	// Joins which turn by at most da are generated as miter joins (m_NumArcPoints = 0): the miter point is at most
	// r / cos(da / 2) - r = tesselation tolerance away from the arc (see the calculation of da), and a miter join has
	// half the geometry of the smallest round join. This is the common case for flattened curves.
	const float cosDa = bx::cos(da);

	RoundJoinArc* arcs = stroker->m_JoinBuffer;
	uint32_t total = 0;
	for (uint32_t iJoin = firstJoin; iJoin < lastJoin; ++iJoin) {
		const Vec2 d01 = dirs[iJoin == 0 ? numDirs - 1 : iJoin - 1];
		const Vec2 d12 = dirs[iJoin];
		const Vec2 v_s = vec2Scale(ext[iJoin], projScale);

		RoundJoinArc* arc = &arcs[iJoin];
		if (vec2Dot(d01, d12) >= cosDa) {
			arc->m_NumArcPoints = 0;
			continue;
		}
		const float leftPointProjDist = d12.x * v_s.x + d12.y * v_s.y;
		if (leftPointProjDist >= 0.0f) {
			// The left point is the inner corner. CCW angle from r01 to r12 in [0, 2*Pi)
			const Vec2 arcDir = vec2ArcDir(vec2PerpCW(d01));
			const Vec2 arcEndDir = vec2ArcDir(vec2PerpCW(d12));
			float arcAngle = bx::atan2(vec2Cross(arcDir, arcEndDir), vec2Dot(arcDir, arcEndDir));
			if (arcAngle < 0.0f) {
				arcAngle += bx::kPi2;
			}

			const uint32_t numArcPoints = bx::max(2u, (uint32_t)(arcAngle / da));
			const float arcDa = arcAngle / (float)numArcPoints;
			arc->m_ArcDir = arcDir;
			arc->m_CosDa = bx::cos(arcDa);
			arc->m_SinDa = bx::sin(arcDa);
			arc->m_NumArcPoints = numArcPoints;
		} else {
			// The right point is the inner corner. CW angle from l01 to l12 in (-2*Pi, 0]
			const Vec2 arcDir = vec2ArcDir(vec2PerpCCW(d01));
			const Vec2 arcEndDir = vec2ArcDir(vec2PerpCCW(d12));
			float arcAngle = bx::atan2(vec2Cross(arcDir, arcEndDir), vec2Dot(arcDir, arcEndDir));
			if (arcAngle > 0.0f) {
				arcAngle -= bx::kPi2;
			}

			const uint32_t numArcPoints = bx::max(2u, (uint32_t)(-arcAngle / da));
			const float arcDa = arcAngle / (float)numArcPoints;
			arc->m_ArcDir = arcDir;
			arc->m_CosDa = bx::cos(arcDa);
			arc->m_SinDa = bx::sin(arcDa);
			arc->m_NumArcPoints = numArcPoints;
		}

		total += arc->m_NumArcPoints;
	}

	*totalArcPoints = total;
	return arcs;
}

Stroker* createStroker(bx::AllocatorI* allocator)
{
	Stroker* stroker = (Stroker*)bx::alloc(allocator, sizeof(Stroker));
	bx::memSet(stroker, 0, sizeof(Stroker));
	stroker->m_Allocator = allocator;
	stroker->m_FringeWidth = 1.0f;
	stroker->m_Scale = 1.0f;
	stroker->m_TesselationTolerance = 0.25f;
	return stroker;
}

void destroyStroker(Stroker* stroker)
{
	bx::AllocatorI* allocator = stroker->m_Allocator;

    if (stroker->m_PosBuffer) {
        bx::alignedFree(allocator, stroker->m_PosBuffer, 16);
    }
    
    if (stroker->m_ColorBuffer) {
        bx::alignedFree(allocator, stroker->m_ColorBuffer, 16);
    }
    
    if (stroker->m_IndexBuffer) {
        bx::alignedFree(allocator, stroker->m_IndexBuffer, 16);
    }

	if (stroker->m_FanIndexBuffer) {
		bx::alignedFree(allocator, stroker->m_FanIndexBuffer, 16);
	}

	if (stroker->m_SegmentBuffer) {
		bx::alignedFree(allocator, stroker->m_SegmentBuffer, 16);
	}

	if (stroker->m_JoinBuffer) {
		bx::alignedFree(allocator, stroker->m_JoinBuffer, 16);
	}

	if (stroker->m_VertexMap) {
		bx::alignedFree(allocator, stroker->m_VertexMap, 16);
	}

	if (stroker->m_EdgeHash) {
		bx::alignedFree(allocator, stroker->m_EdgeHash, 16);
	}

	if (stroker->m_ClampScratch) {
		bx::alignedFree(allocator, stroker->m_ClampScratch, 16);
	}

	if (stroker->m_ContourVertices) {
		bx::alignedFree(allocator, stroker->m_ContourVertices, 16);
	}

	if (stroker->m_ContourSizes) {
		bx::alignedFree(allocator, stroker->m_ContourSizes, 16);
	}

	if (stroker->m_Tesselator) {
		tessDeleteTess(stroker->m_Tesselator);
	}

	libtess2Reset(&stroker->m_libTessAllocator);

	if (stroker->m_libTessAllocator.m_Buffer) {
		bx::alignedFree(allocator, stroker->m_libTessAllocator.m_Buffer, 16);
	}

	bx::free(allocator, stroker);
}

void strokerReset(Stroker* stroker, float scale, float tesselationTolerance, float fringeWidth)
{
	stroker->m_Scale = scale;
	stroker->m_TesselationTolerance = tesselationTolerance;
	stroker->m_FringeWidth = fringeWidth;
}

void strokerPolylineStroke(Stroker* stroker, Mesh* mesh, const float* vertexList, uint32_t numPathVertices, bool isClosed, float strokeWidth, LineCap::Enum lineCap, LineJoin::Enum lineJoin, const StrokerSink* sink)
{
	const uint8_t perm = (((uint8_t)lineCap) << 1)
		| (((uint8_t)lineJoin) << 3)
		| (isClosed ? 0x01 : 0x00);

	const Vec2* vtx = (const Vec2*)vertexList;

	switch (perm) {
	case  0: polylineStroke<false, LineCap::Butt, LineJoin::Miter>(stroker, mesh, vtx, numPathVertices, strokeWidth, sink);   break;
	case  1: polylineStroke<true, LineCap::Butt, LineJoin::Miter>(stroker, mesh, vtx, numPathVertices, strokeWidth, sink);    break;
	case  2: polylineStroke<false, LineCap::Round, LineJoin::Miter>(stroker, mesh, vtx, numPathVertices, strokeWidth, sink);  break;
	case  3: polylineStroke<true, LineCap::Butt, LineJoin::Miter>(stroker, mesh, vtx, numPathVertices, strokeWidth, sink);    break;
	case  4: polylineStroke<false, LineCap::Square, LineJoin::Miter>(stroker, mesh, vtx, numPathVertices, strokeWidth, sink); break;
	case  5: polylineStroke<true, LineCap::Butt, LineJoin::Miter>(stroker, mesh, vtx, numPathVertices, strokeWidth, sink);    break;
		// 6 to 7 == invalid line cap type
	case  8: polylineStroke<false, LineCap::Butt, LineJoin::Round>(stroker, mesh, vtx, numPathVertices, strokeWidth, sink);   break;
	case  9: polylineStroke<true, LineCap::Butt, LineJoin::Round>(stroker, mesh, vtx, numPathVertices, strokeWidth, sink);    break;
	case 10: polylineStroke<false, LineCap::Round, LineJoin::Round>(stroker, mesh, vtx, numPathVertices, strokeWidth, sink);  break;
	case 11: polylineStroke<true, LineCap::Butt, LineJoin::Round>(stroker, mesh, vtx, numPathVertices, strokeWidth, sink);    break;
	case 12: polylineStroke<false, LineCap::Square, LineJoin::Round>(stroker, mesh, vtx, numPathVertices, strokeWidth, sink); break;
	case 13: polylineStroke<true, LineCap::Butt, LineJoin::Round>(stroker, mesh, vtx, numPathVertices, strokeWidth, sink);    break;
		// 14 to 15 == invalid line cap type
	case 16: polylineStroke<false, LineCap::Butt, LineJoin::Bevel>(stroker, mesh, vtx, numPathVertices, strokeWidth, sink);   break;
	case 17: polylineStroke<true, LineCap::Butt, LineJoin::Bevel>(stroker, mesh, vtx, numPathVertices, strokeWidth, sink);    break;
	case 18: polylineStroke<false, LineCap::Round, LineJoin::Bevel>(stroker, mesh, vtx, numPathVertices, strokeWidth, sink);  break;
	case 19: polylineStroke<true, LineCap::Butt, LineJoin::Bevel>(stroker, mesh, vtx, numPathVertices, strokeWidth, sink);    break;
	case 20: polylineStroke<false, LineCap::Square, LineJoin::Bevel>(stroker, mesh, vtx, numPathVertices, strokeWidth, sink); break;
	case 21: polylineStroke<true, LineCap::Butt, LineJoin::Bevel>(stroker, mesh, vtx, numPathVertices, strokeWidth, sink);    break;
		// 22 to 32 == invalid line join type
	default:
		VG_WARN(false, "Invalid stroke configuration");
		break;
	}
}

void strokerPolylineStrokeAA(Stroker* stroker, Mesh* mesh, const float* vertexList, uint32_t numPathVertices, bool isClosed, Color color, float strokeWidth, LineCap::Enum lineCap, LineJoin::Enum lineJoin, const StrokerSink* sink)
{
	const uint8_t perm = (((uint8_t)lineCap) << 1)
		| (((uint8_t)lineJoin) << 3)
		| (isClosed ? 0x01 : 0x00);

	const Vec2* vtx = (const Vec2*)vertexList;

	switch (perm) {
	case  0: polylineStrokeAA<false, LineCap::Butt, LineJoin::Miter>(stroker, mesh, vtx, numPathVertices, strokeWidth, color, sink);   break;
	case  1: polylineStrokeAA<true, LineCap::Butt, LineJoin::Miter>(stroker, mesh, vtx, numPathVertices, strokeWidth, color, sink);    break;
	case  2: polylineStrokeAA<false, LineCap::Round, LineJoin::Miter>(stroker, mesh, vtx, numPathVertices, strokeWidth, color, sink);  break;
	case  3: polylineStrokeAA<true, LineCap::Butt, LineJoin::Miter>(stroker, mesh, vtx, numPathVertices, strokeWidth, color, sink);    break;
	case  4: polylineStrokeAA<false, LineCap::Square, LineJoin::Miter>(stroker, mesh, vtx, numPathVertices, strokeWidth, color, sink); break;
	case  5: polylineStrokeAA<true, LineCap::Butt, LineJoin::Miter>(stroker, mesh, vtx, numPathVertices, strokeWidth, color, sink);    break;
		// 6 to 7 == invalid line cap type
	case  8: polylineStrokeAA<false, LineCap::Butt, LineJoin::Round>(stroker, mesh, vtx, numPathVertices, strokeWidth, color, sink);   break;
	case  9: polylineStrokeAA<true, LineCap::Butt, LineJoin::Round>(stroker, mesh, vtx, numPathVertices, strokeWidth, color, sink);    break;
	case 10: polylineStrokeAA<false, LineCap::Round, LineJoin::Round>(stroker, mesh, vtx, numPathVertices, strokeWidth, color, sink);  break;
	case 11: polylineStrokeAA<true, LineCap::Butt, LineJoin::Round>(stroker, mesh, vtx, numPathVertices, strokeWidth, color, sink);    break;
	case 12: polylineStrokeAA<false, LineCap::Square, LineJoin::Round>(stroker, mesh, vtx, numPathVertices, strokeWidth, color, sink); break;
	case 13: polylineStrokeAA<true, LineCap::Butt, LineJoin::Round>(stroker, mesh, vtx, numPathVertices, strokeWidth, color, sink);    break;
		// 14 to 15 == invalid line cap type
	case 16: polylineStrokeAA<false, LineCap::Butt, LineJoin::Bevel>(stroker, mesh, vtx, numPathVertices, strokeWidth, color, sink);   break;
	case 17: polylineStrokeAA<true, LineCap::Butt, LineJoin::Bevel>(stroker, mesh, vtx, numPathVertices, strokeWidth, color, sink);    break;
	case 18: polylineStrokeAA<false, LineCap::Round, LineJoin::Bevel>(stroker, mesh, vtx, numPathVertices, strokeWidth, color, sink);  break;
	case 19: polylineStrokeAA<true, LineCap::Butt, LineJoin::Bevel>(stroker, mesh, vtx, numPathVertices, strokeWidth, color, sink);    break;
	case 20: polylineStrokeAA<false, LineCap::Square, LineJoin::Bevel>(stroker, mesh, vtx, numPathVertices, strokeWidth, color, sink); break;
	case 21: polylineStrokeAA<true, LineCap::Butt, LineJoin::Bevel>(stroker, mesh, vtx, numPathVertices, strokeWidth, color, sink);    break;
		// 22 to 32 == invalid line join type
	default:
		VG_WARN(false, "Invalid stroke configuration");
		break;
	}
}

void strokerPolylineStrokeAAThin(Stroker* stroker, Mesh* mesh, const float* vertexList, uint32_t numPathVertices, bool isClosed, Color color, LineCap::Enum lineCap, LineJoin::Enum lineJoin, const StrokerSink* sink)
{
	// TODO: Why is isClosed passed as argument instead of template param?
	const uint8_t perm = ((uint8_t)lineCap) | (((uint8_t)lineJoin) << 2);

	const Vec2* vtx = (const Vec2*)vertexList;

	switch (perm) {
	case  0: polylineStrokeAAThin<LineCap::Butt, LineJoin::Miter>(stroker, mesh, vtx, numPathVertices, color, isClosed, sink);   break;
	case  1: polylineStrokeAAThin<LineCap::Square, LineJoin::Miter>(stroker, mesh, vtx, numPathVertices, color, isClosed, sink);   break;
	case  2: polylineStrokeAAThin<LineCap::Square, LineJoin::Miter>(stroker, mesh, vtx, numPathVertices, color, isClosed, sink);   break;
	case  4: polylineStrokeAAThin<LineCap::Butt, LineJoin::Bevel>(stroker, mesh, vtx, numPathVertices, color, isClosed, sink);   break;
	case  5: polylineStrokeAAThin<LineCap::Square, LineJoin::Bevel>(stroker, mesh, vtx, numPathVertices, color, isClosed, sink);   break;
	case  6: polylineStrokeAAThin<LineCap::Square, LineJoin::Bevel>(stroker, mesh, vtx, numPathVertices, color, isClosed, sink);   break;
	case  8: polylineStrokeAAThin<LineCap::Butt, LineJoin::Bevel>(stroker, mesh, vtx, numPathVertices, color, isClosed, sink);   break;
	case  9: polylineStrokeAAThin<LineCap::Square, LineJoin::Bevel>(stroker, mesh, vtx, numPathVertices, color, isClosed, sink);   break;
	case 10: polylineStrokeAAThin<LineCap::Square, LineJoin::Bevel>(stroker, mesh, vtx, numPathVertices, color, isClosed, sink);   break;
	default:
		VG_WARN(false, "Invalid stroke configuration");
		break;
	}
}

void strokerConvexFill(Stroker* stroker, Mesh* mesh, const float* vertexList, uint32_t numVertices)
{
	const uint32_t numTris = numVertices - 2;
	const uint32_t numIndices = numTris * 3; // N - 2 triangles in a N pt fan, 3 indices per triangle

	resetGeometry(stroker);

	// The fan indices only depend on the number of vertices, so they are kept in a separate grow-only
	// buffer and only the missing triangles are generated when a bigger polygon than before shows up.
	if (numTris > stroker->m_FanTriCapacity) {
		const uint32_t oldCapacity = stroker->m_FanTriCapacity;
		const uint32_t newCapacity = bx::max<uint32_t>(numTris, oldCapacity + (oldCapacity >> 1));
		stroker->m_FanIndexBuffer = (uint16_t*)bx::alignedRealloc(stroker->m_Allocator, stroker->m_FanIndexBuffer, sizeof(uint16_t) * 3 * newCapacity, 16);
		stroker->m_FanTriCapacity = newCapacity;

		uint16_t* dstIndex = &stroker->m_FanIndexBuffer[oldCapacity * 3];
		uint16_t nextID = (uint16_t)(oldCapacity + 1);
		for (uint32_t i = oldCapacity; i < newCapacity; ++i) {
			*dstIndex++ = 0;
			*dstIndex++ = nextID;
			*dstIndex++ = nextID + 1;

			++nextID;
		}
	}

	mesh->m_PosBuffer = vertexList;
	mesh->m_ColorBuffer = nullptr;
	mesh->m_IndexBuffer = stroker->m_FanIndexBuffer;
	mesh->m_NumVertices = numVertices;
	mesh->m_NumIndices = numIndices;
}

#if VG_CONFIG_ENABLE_SIMD && BX_CPU_X86
void strokerConvexFillAA(Stroker* stroker, Mesh* mesh, const float* vertexList, uint32_t numVertices, uint32_t color, const StrokerSink* sink)
{
	VG_CHECK(numVertices >= 3, "Invalid number of vertices");

	const uint32_t lastVertexID = numVertices - 1;

	const __m128 vtx0 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)vertexList);
	const __m128 vtx1 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(vertexList + 2));
	const __m128 vtx2 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(vertexList + 4));
	const float cross = xmm_vec2_cross(_mm_sub_ps(vtx1, vtx0), _mm_sub_ps(vtx2, vtx0));

	const float aa = stroker->m_FringeWidth * 0.5f * bx::sign(cross);
	const __m128 xmm_aa = _mm_set_ps1(aa);

	const uint32_t c0 = colorSetAlpha(color, 0);

	const uint32_t numTris =
		(numVertices - 2) + // Triangle fan
		(numVertices * 2); // AA fringes
	const uint32_t numDrawVertices = numVertices * 2; // original polygon point + AA fringe point.
	const uint32_t numDrawIndices = numTris * 3;

	GeometryOutput out;
	beginGeometry(stroker, sink, numDrawVertices, numDrawIndices, true, &out);

	// Vertex buffer
	{
		const __m128 vtxLast = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)(vertexList + (lastVertexID << 1)));
		__m128 d01 = xmm_vec2_dir(vtxLast, vtx0);
		__m128 p1 = vtx0;

		const float* srcPos = vertexList + 2;
		float* dstPos = &out.m_Pos->x;

		const __m128 xmm_epsilon = _mm_set_ps1(VG_EPSILON);
		const __m128 xmm_minExtrusionCross = _mm_set_ps1(kMinExtrusionCross);
		const __m128 xmm_absMask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
		const __m128 vec2x2_perpCCW_xorMask = _mm_castsi128_ps(_mm_set_epi32(0, 0x80000000, 0, 0x80000000));

		const uint32_t numIter = lastVertexID >> 2;
		for (uint32_t i = 0; i < numIter; ++i) {
			// Load 4 points. With p1 from previous loop iteration make up 4 segments
			const __m128 p23 = _mm_loadu_ps(srcPos);                          // { p2.x, p2.y, p3.x, p3.y }
			const __m128 p45 = _mm_loadu_ps(srcPos + 4);                      // { p4.x, p4.y, p5.x, p5.y }

			const __m128 p12 = _mm_movelh_ps(p1, p23);                        // { p1.x, p1.y, p2.x, p2.y }
			const __m128 p34 = _mm_movelh_ps(_mm_movehl_ps(p23, p23), p45);   // { p3.x, p3.y, p4.x, p4.y }

			// Calculate the direction vector of the 4 segments
			// NOTE: Tried to calc all 4 rsqrt in 1 call but it ends up being slower. Kept this version for now.
			const __m128 d12_23_unorm = _mm_sub_ps(p23, p12);
			const __m128 d34_45_unorm = _mm_sub_ps(p45, p34);

			const __m128 d12_23_xy_sqr = _mm_mul_ps(d12_23_unorm, d12_23_unorm);
			const __m128 d34_45_xy_sqr = _mm_mul_ps(d34_45_unorm, d34_45_unorm);

			const __m128 d12_23_yx_sqr = _mm_shuffle_ps(d12_23_xy_sqr, d12_23_xy_sqr, _MM_SHUFFLE(2, 3, 0, 1));
			const __m128 d34_45_yx_sqr = _mm_shuffle_ps(d34_45_xy_sqr, d34_45_xy_sqr, _MM_SHUFFLE(2, 3, 0, 1));

			const __m128 len12_23_sqr = _mm_add_ps(d12_23_xy_sqr, d12_23_yx_sqr);
			const __m128 len34_45_sqr = _mm_add_ps(d34_45_xy_sqr, d34_45_yx_sqr);

			const __m128 lenSqr123_ge_eps = _mm_cmpge_ps(len12_23_sqr, xmm_epsilon);
			const __m128 lenSqr345_ge_eps = _mm_cmpge_ps(len34_45_sqr, xmm_epsilon);

			const __m128 invLen12_23 = xmm_rsqrt(len12_23_sqr);
			const __m128 invLen34_45 = xmm_rsqrt(len34_45_sqr);

			const __m128 invLen12_23_masked = _mm_and_ps(invLen12_23, lenSqr123_ge_eps);
			const __m128 invLen34_45_masked = _mm_and_ps(invLen34_45, lenSqr345_ge_eps);

			const __m128 d12_23 = _mm_mul_ps(d12_23_unorm, invLen12_23_masked);
			const __m128 d34_45 = _mm_mul_ps(d34_45_unorm, invLen34_45_masked);

			// Calculate the 4 extrusion vectors for the 4 points based on the equ
			// abs(cross(d12, d01)) > kMinExtrusionCross ? ((d01 - d12) / cross(d12, d01)) : rot90CCW(d01)
			const __m128 v012_123_fake = _mm_xor_ps(_mm_shuffle_ps(d01, d12_23, _MM_SHUFFLE(0, 1, 0, 1)), vec2x2_perpCCW_xorMask);
			const __m128 v234_345_fake = _mm_xor_ps(_mm_shuffle_ps(d12_23, d34_45, _MM_SHUFFLE(0, 1, 2, 3)), vec2x2_perpCCW_xorMask);

			// cross012 = d12.x * d01.y - d12.y * d01.x
			// cross123 = d23.x * d12.y - d23.y * d12.x
			// cross234 = d34.x * d23.y - d34.y * d23.x
			// cross345 = d45.x * d34.y - d45.y * d34.x
			const __m128 dxy01_12 = _mm_shuffle_ps(d01, d12_23, _MM_SHUFFLE(1, 0, 1, 0));
			const __m128 dxy12_23 = d12_23;
			const __m128 dxy23_34 = _mm_shuffle_ps(d12_23, d34_45, _MM_SHUFFLE(1, 0, 3, 2));
			const __m128 dxy34_45 = d34_45;

			const __m128 dx01_12_23_34 = _mm_shuffle_ps(dxy01_12, dxy23_34, _MM_SHUFFLE(2, 0, 2, 0));
			const __m128 dy01_12_23_34 = _mm_shuffle_ps(dxy01_12, dxy23_34, _MM_SHUFFLE(3, 1, 3, 1));
			const __m128 dx12_23_34_45 = _mm_shuffle_ps(dxy12_23, dxy34_45, _MM_SHUFFLE(2, 0, 2, 0));
			const __m128 dy12_23_34_45 = _mm_shuffle_ps(dxy12_23, dxy34_45, _MM_SHUFFLE(3, 1, 3, 1));

			const __m128 crossx012_123_234_345 = _mm_mul_ps(dx12_23_34_45, dy01_12_23_34);
			const __m128 crossy012_123_234_345 = _mm_mul_ps(dy12_23_34_45, dx01_12_23_34);

			const __m128 cross012_123_234_345 = _mm_sub_ps(crossx012_123_234_345, crossy012_123_234_345);

			const __m128 inv_cross012_123_234_345 = xmm_rcp(cross012_123_234_345);

			const __m128 cross_gt_eps012_123_234_345 = _mm_cmpgt_ps(_mm_and_ps(cross012_123_234_345, xmm_absMask), xmm_minExtrusionCross);

			const __m128 inv_cross012_123 = _mm_shuffle_ps(inv_cross012_123_234_345, inv_cross012_123_234_345, _MM_SHUFFLE(1, 1, 0, 0));
			const __m128 inv_cross234_345 = _mm_shuffle_ps(inv_cross012_123_234_345, inv_cross012_123_234_345, _MM_SHUFFLE(3, 3, 2, 2));

			const __m128 cross012_123_gt_eps = _mm_shuffle_ps(cross_gt_eps012_123_234_345, cross_gt_eps012_123_234_345, _MM_SHUFFLE(1, 1, 0, 0));
			const __m128 cross234_345_gt_eps = _mm_shuffle_ps(cross_gt_eps012_123_234_345, cross_gt_eps012_123_234_345, _MM_SHUFFLE(3, 3, 2, 2));

			const __m128 dxy012_123 = _mm_sub_ps(dxy01_12, dxy12_23);
			const __m128 dxy234_345 = _mm_sub_ps(dxy23_34, dxy34_45);

			const __m128 v012_123_true = _mm_mul_ps(dxy012_123, inv_cross012_123);
			const __m128 v234_345_true = _mm_mul_ps(dxy234_345, inv_cross234_345);

			const __m128 v012_123_true_masked = _mm_and_ps(cross012_123_gt_eps, v012_123_true);
			const __m128 v234_345_true_masked = _mm_and_ps(cross234_345_gt_eps, v234_345_true);

			const __m128 v012_123_fake_masked = _mm_andnot_ps(cross012_123_gt_eps, v012_123_fake);
			const __m128 v245_345_fake_masked = _mm_andnot_ps(cross234_345_gt_eps, v234_345_fake);

			const __m128 v012_123 = _mm_or_ps(v012_123_true_masked, v012_123_fake_masked);
			const __m128 v234_345 = _mm_or_ps(v234_345_true_masked, v245_345_fake_masked);

			const __m128 v012_v123_aa = _mm_mul_ps(v012_123, xmm_aa);
			const __m128 v234_v345_aa = _mm_mul_ps(v234_345, xmm_aa);

			// Calculate the 2 fringe points for each of p1, p2, p3 and p4
			const __m128 posEdge12 = _mm_add_ps(p12, v012_v123_aa);
			const __m128 negEdge12 = _mm_sub_ps(p12, v012_v123_aa);
			const __m128 posEdge34 = _mm_add_ps(p34, v234_v345_aa);
			const __m128 negEdge34 = _mm_sub_ps(p34, v234_v345_aa);

			const __m128 p1_in_out = _mm_shuffle_ps(posEdge12, negEdge12, _MM_SHUFFLE(1, 0, 1, 0));
			const __m128 p2_in_out = _mm_shuffle_ps(posEdge12, negEdge12, _MM_SHUFFLE(3, 2, 3, 2));
			const __m128 p3_in_out = _mm_shuffle_ps(posEdge34, negEdge34, _MM_SHUFFLE(1, 0, 1, 0));
			const __m128 p4_in_out = _mm_shuffle_ps(posEdge34, negEdge34, _MM_SHUFFLE(3, 2, 3, 2));

			// Store the fringe points
			_mm_storeu_ps(dstPos + 0, p1_in_out);
			_mm_storeu_ps(dstPos + 4, p2_in_out);
			_mm_storeu_ps(dstPos + 8, p3_in_out);
			_mm_storeu_ps(dstPos + 12, p4_in_out);

			// Move on to the next iteration.
			d01 = _mm_movehl_ps(d34_45, d34_45);
			p1 = _mm_movehl_ps(p45, p45); // p1 = p5
			srcPos += 8;
			dstPos += 16;
		}

		uint32_t rem = (lastVertexID & 3);
		if (rem >= 2) {
			const __m128 p23 = _mm_loadu_ps(srcPos);
			const __m128 p12 = _mm_movelh_ps(p1, p23);

			const __m128 d12_23 = _mm_sub_ps(p23, p12);
			const __m128 d12_23_xy_sqr = _mm_mul_ps(d12_23, d12_23);
			const __m128 d12_23_yx_sqr = _mm_shuffle_ps(d12_23_xy_sqr, d12_23_xy_sqr, _MM_SHUFFLE(2, 3, 0, 1));
			const __m128 len12_23_sqr = _mm_add_ps(d12_23_xy_sqr, d12_23_yx_sqr);
			const __m128 lenSqr_ge_eps = _mm_cmpge_ps(len12_23_sqr, xmm_epsilon);

			const __m128 invLen12_23 = xmm_rsqrt(len12_23_sqr);

			const __m128 invLen12_23_masked = _mm_and_ps(invLen12_23, lenSqr_ge_eps);
			const __m128 d12_23_norm = _mm_mul_ps(d12_23, invLen12_23_masked);

			const __m128 d12 = _mm_movelh_ps(d12_23_norm, d12_23_norm);
			const __m128 d23 = _mm_movehl_ps(d12_23_norm, d12_23_norm);

			const __m128 d12xy_d01xy = _mm_movelh_ps(d12, d01);
			const __m128 d23xy_d12xy = _mm_movelh_ps(d23, d12);

			const __m128 d01yx_d12yx = _mm_shuffle_ps(d12xy_d01xy, d12xy_d01xy, _MM_SHUFFLE(0, 1, 2, 3));
			const __m128 d12yx_d23yx = _mm_shuffle_ps(d23xy_d12xy, d23xy_d12xy, _MM_SHUFFLE(0, 1, 2, 3));

			const __m128 d12xd01y_d12yd01x = _mm_mul_ps(d12xy_d01xy, d01yx_d12yx);
			const __m128 d23xd12y_d23yd12x = _mm_mul_ps(d23xy_d12xy, d12yx_d23yx);

			const __m128 d12yd01x_d23yd12x = _mm_shuffle_ps(d12xd01y_d12yd01x, d23xd12y_d23yd12x, _MM_SHUFFLE(1, 1, 1, 1));
			const __m128 d12xd01y_d23xd12x = _mm_shuffle_ps(d12xd01y_d12yd01x, d23xd12y_d23yd12x, _MM_SHUFFLE(0, 0, 0, 0));

			const __m128 cross012_123 = _mm_sub_ps(d12xd01y_d23xd12x, d12yd01x_d23yd12x);

			const __m128 inv_cross012_123 = xmm_rcp(cross012_123);

			const __m128 v012_123_fake = _mm_xor_ps(d01yx_d12yx, vec2x2_perpCCW_xorMask);

			const __m128 d01xy_d12xy = _mm_shuffle_ps(d12xy_d01xy, d12xy_d01xy, _MM_SHUFFLE(1, 0, 3, 2));
			const __m128 d12xy_d23xy = _mm_shuffle_ps(d23xy_d12xy, d23xy_d12xy, _MM_SHUFFLE(1, 0, 3, 2));

			const __m128 d012xy_d123xy = _mm_sub_ps(d01xy_d12xy, d12xy_d23xy);
			const __m128 v012_123_true = _mm_mul_ps(d012xy_d123xy, inv_cross012_123);

			const __m128 cross_gt_eps = _mm_cmpgt_ps(_mm_and_ps(cross012_123, xmm_absMask), xmm_minExtrusionCross);
			const __m128 v012_123_true_masked = _mm_and_ps(cross_gt_eps, v012_123_true);
			const __m128 v012_123_fake_masked = _mm_andnot_ps(cross_gt_eps, v012_123_fake);
			const __m128 v012_123 = _mm_or_ps(v012_123_true_masked, v012_123_fake_masked);

			const __m128 v012_v123_aa = _mm_mul_ps(v012_123, xmm_aa);

			const __m128 posEdge = _mm_add_ps(p12, v012_v123_aa);
			const __m128 negEdge = _mm_sub_ps(p12, v012_v123_aa);

			const __m128 packed0 = _mm_shuffle_ps(posEdge, negEdge, _MM_SHUFFLE(1, 0, 1, 0));
			const __m128 packed1 = _mm_shuffle_ps(posEdge, negEdge, _MM_SHUFFLE(3, 2, 3, 2));

			_mm_storeu_ps(dstPos, packed0);
			_mm_storeu_ps(dstPos + 4, packed1);

			dstPos += 8;
			srcPos += 4;
			d01 = d23;
			p1 = _mm_movehl_ps(p23, p23);

			rem -= 2;
		}

		if (rem) {
			const __m128 p2 = _mm_loadl_pi(_mm_setzero_ps(), (const __m64*)srcPos);
			const __m128 d12 = xmm_vec2_dir(p1, p2);
			const __m128 v_aa = _mm_mul_ps(xmm_calcExtrusionVector(d01, d12), xmm_aa);
			const __m128 packed = _mm_movelh_ps(_mm_add_ps(p1, v_aa), _mm_sub_ps(p1, v_aa));
			_mm_storeu_ps(dstPos, packed);

			dstPos += 4;
			srcPos += 2;
			d01 = d12;
			p1 = p2;
		}

		// Last segment
		{
			const __m128 v_aa = _mm_mul_ps(xmm_calcExtrusionVector(d01, xmm_vec2_dir(p1, vtx0)), xmm_aa);
			const __m128 packed = _mm_movelh_ps(_mm_add_ps(p1, v_aa), _mm_sub_ps(p1, v_aa));
			_mm_storeu_ps(dstPos, packed);
		}

		const uint32_t colors[2] = { color, c0 };
		vgutil::memset64(out.m_Color, numVertices, &colors[0]);

	}

	// Index buffer
	{
		uint16_t* dstIndex = out.m_Index;

		// First fringe quad
		dstIndex[0] = 0; dstIndex[1] = 1; dstIndex[2] = 3;
		dstIndex[3] = 0; dstIndex[4] = 3; dstIndex[5] = 2;
		dstIndex += 6;

		const uint32_t numFanTris = numVertices - 2;

		__m128i xmm_stv = _mm_set1_epi16(2);
		{
			static const uint16_t delta0[8] = { 0, 0, 2, 0, 1, 3, 0, 3 };
			static const uint16_t delta1[8] = { 2, 0, 2, 4, 2, 3, 5, 2 };
			static const uint16_t delta2[8] = { 5, 4, 0, 4, 6, 4, 5, 7 };
			static const uint16_t delta3[8] = { 4, 7, 6, 0, 6, 8, 6, 7 };
			static const uint16_t delta4[8] = { 9, 6, 9, 8, 0, 0, 0, 0 };
			const __m128i xmm_delta0 = _mm_loadu_si128((const __m128i*)delta0);
			const __m128i xmm_delta1 = _mm_loadu_si128((const __m128i*)delta1);
			const __m128i xmm_delta2 = _mm_loadu_si128((const __m128i*)delta2);
			const __m128i xmm_delta3 = _mm_loadu_si128((const __m128i*)delta3);
			const __m128i xmm_delta4 = _mm_loadu_si128((const __m128i*)delta4);

			const __m128i xmm_stv_delta = _mm_set1_epi16(8);

			const uint32_t numIter = numFanTris >> 2;
			for (uint32_t i = 0; i < numIter; ++i) {
				// { 0, stv + 0, stv + 2, stv + 0, stv + 1, stv + 3, stv + 0, stv + 3 }
				// { stv + 2, 0, stv + 2, stv + 4, stv + 2, stv + 3, stv + 5, stv + 2 }
				// { stv + 5, stv + 4, 0, stv + 4, stv + 6, stv + 4, stv + 5, stv + 7 }
				// { stv + 4, stv + 7, stv + 6, 0, stv + 6, stv + 8, stv + 6, stv + 7 }
				// { stv + 9, stv + 6, stv + 9, stv + 8 }
				const __m128i xmm_id0 = _mm_add_epi16(xmm_stv, xmm_delta0);
				const __m128i xmm_id1 = _mm_add_epi16(xmm_stv, xmm_delta1);
				const __m128i xmm_id2 = _mm_add_epi16(xmm_stv, xmm_delta2);
				const __m128i xmm_id3 = _mm_add_epi16(xmm_stv, xmm_delta3);
				const __m128i xmm_id4 = _mm_add_epi16(xmm_stv, xmm_delta4);

				_mm_storeu_si128((__m128i*)(dstIndex + 0), _mm_insert_epi16(xmm_id0, 0, 0));
				_mm_storeu_si128((__m128i*)(dstIndex + 8), _mm_insert_epi16(xmm_id1, 0, 1));
				_mm_storeu_si128((__m128i*)(dstIndex + 16), _mm_insert_epi16(xmm_id2, 0, 2));
				_mm_storeu_si128((__m128i*)(dstIndex + 24), _mm_insert_epi16(xmm_id3, 0, 3));
				_mm_storel_epi64((__m128i*)(dstIndex + 32), xmm_id4);

				dstIndex += 36;
				xmm_stv = _mm_add_epi16(xmm_stv, xmm_stv_delta);
			}
		}

		{
			static const uint16_t delta0[8] = { 0, 2, 0, 1, 3, 0, 3, 2 };
			static const uint16_t delta1[8] = { 2, 4, 2, 3, 5, 2, 5, 4 };
			const __m128i xmm_delta0 = _mm_loadu_si128((const __m128i*)delta0);

			uint32_t rem = numFanTris & 3;
			if (rem >= 2) {
				const __m128i xmm_delta1 = _mm_loadu_si128((const __m128i*)delta1);
				const __m128i xmm_id0 = _mm_add_epi16(xmm_stv, xmm_delta0);
				const __m128i xmm_id1 = _mm_add_epi16(xmm_stv, xmm_delta1);

				dstIndex[0] = 0;
				_mm_storeu_si128((__m128i*)(dstIndex + 1), xmm_id0);

				dstIndex[9] = 0;
				_mm_storeu_si128((__m128i*)(dstIndex + 10), xmm_id1);

				dstIndex += 18;
				xmm_stv = _mm_add_epi16(xmm_stv, _mm_set1_epi16(4));
				rem -= 2;
			}

			if (rem) {
				const __m128i xmm_id0 = _mm_add_epi16(xmm_stv, xmm_delta0);

				dstIndex[0] = 0;
				_mm_storeu_si128((__m128i*)(dstIndex + 1), xmm_id0);

				dstIndex += 9;
			}
		}

		// Last fringe quad
		const uint16_t lastID = (uint16_t)((numVertices - 1) << 1);
		dstIndex[0] = lastID;
		dstIndex[1] = lastID + 1;
		dstIndex[2] = 1;
		dstIndex[3] = lastID;
		dstIndex[4] = 1;
		dstIndex[5] = 0;

	}

	endGeometry(stroker, &out, out.m_Pos + numDrawVertices, out.m_Index + numDrawIndices, true, mesh);
}
#else
void strokerConvexFillAA(Stroker* stroker, Mesh* mesh, const float* vertexList, uint32_t numVertices, uint32_t color, const StrokerSink* sink)
{
	// Determine path orientation by checking the normal of the first triangle
	// WARNING: Might not work in all cases.
	VG_CHECK(numVertices >= 3, "Invalid number of vertices");

	const Vec2* vtx = (const Vec2*)vertexList;

	const float cross = vec2Cross(vec2Sub(vtx[1], vtx[0]), vec2Sub(vtx[2], vtx[0]));

	const float aa = stroker->m_FringeWidth * 0.5f * bx::sign(cross);
	const uint32_t c0 = colorSetAlpha(color, 0);

	const uint32_t numTris =
		(numVertices - 2) + // Triangle fan
		(numVertices * 2); // AA fringes
	const uint32_t numDrawVertices = numVertices * 2; // original polygon point + AA fringe point.
	const uint32_t numDrawIndices = numTris * 3;

	GeometryOutput out;
	beginGeometry(stroker, sink, numDrawVertices, numDrawIndices, true, &out);

	// Vertex buffer
	{
		Vec2 d01 = vec2Dir(vtx[numVertices - 1], vtx[0]);

		Vec2* dstPos = out.m_Pos;
		for (uint32_t iSegment = 0; iSegment < numVertices; ++iSegment) {
			const Vec2& p1 = vtx[iSegment];
			const Vec2& p2 = vtx[iSegment == numVertices - 1 ? 0 : iSegment + 1];

			const Vec2 d12 = vec2Dir(p1, p2);
			const Vec2 v = calcExtrusionVector(d01, d12);
			const Vec2 v_aa = vec2Scale(v, aa);

			dstPos[0] = vec2Add(p1, v_aa);
			dstPos[1] = vec2Sub(p1, v_aa);
			dstPos += 2;

			d01 = d12;
		}

		const uint32_t colors[2] = { color, c0 };
		vgutil::memset64(out.m_Color, numVertices, &colors[0]);

	}

	// Index buffer
	{
		uint16_t* dstIndex = out.m_Index;

		// Generate the triangle fan (original polygon)
		const uint32_t numFanTris = numVertices - 2;
		uint16_t secondTriVertex = 2;
		for (uint32_t i = 0; i < numFanTris; ++i) {
			*dstIndex++ = 0;
			*dstIndex++ = secondTriVertex;
			*dstIndex++ = secondTriVertex + 2;
			secondTriVertex += 2;
		}

		// Generate the AA fringes
		uint16_t firstVertexID = 0;
		for (uint32_t i = 0; i < numVertices - 1; ++i) {
			*dstIndex++ = firstVertexID;
			*dstIndex++ = firstVertexID + 1;
			*dstIndex++ = firstVertexID + 3;
			*dstIndex++ = firstVertexID;
			*dstIndex++ = firstVertexID + 3;
			*dstIndex++ = firstVertexID + 2;
			firstVertexID += 2;
		}

		// Last segment
		*dstIndex++ = firstVertexID;
		*dstIndex++ = firstVertexID + 1;
		*dstIndex++ = 1;
		*dstIndex++ = firstVertexID;
		*dstIndex++ = 1;
		*dstIndex++ = 0;

	}

	endGeometry(stroker, &out, out.m_Pos + numDrawVertices, out.m_Index + numDrawIndices, true, mesh);
}
#endif

// (Re)creates the tesselator and resets the scratch memory.
static void resetTesselator(Stroker* stroker)
{
	// Delete old tesselator
	if (stroker->m_Tesselator) {
		tessDeleteTess(stroker->m_Tesselator);
	}

#if VG_CONFIG_LIBTESS2_SCRATCH_BUFFER
	// Initialize the allocator once
	if (!stroker->m_libTessAllocator.m_Buffer) {
		stroker->m_libTessAllocator.m_Allocator = stroker->m_Allocator;
		stroker->m_libTessAllocator.m_Capacity = VG_CONFIG_LIBTESS2_SCRATCH_BUFFER;
		stroker->m_libTessAllocator.m_Buffer = (uint8_t*)bx::alignedAlloc(stroker->m_Allocator, stroker->m_libTessAllocator.m_Capacity, 16);
	}

	// Reset the allocator.
	libtess2Reset(&stroker->m_libTessAllocator);

	// Initialize the tesselator
	TESSalloc allocator;
	allocator.meshEdgeBucketSize = 256;
	allocator.meshVertexBucketSize = 256;
	allocator.meshFaceBucketSize = 256;
	allocator.dictNodeBucketSize = 256;
	allocator.regionBucketSize = 256;
	allocator.extraVertices = 256;
	allocator.userData = &stroker->m_libTessAllocator;
	allocator.memalloc = libtess2Alloc;
	allocator.memfree = libtess2Free;
	allocator.memrealloc = libtess2Realloc;
	stroker->m_Tesselator = tessNewTess(&allocator);
#else
	stroker->m_Tesselator = tessNewTess(nullptr);
#endif
}

bool strokerConcaveFillBegin(Stroker* stroker)
{
	// The tesselator is (re)created only when it's used (see addContoursToTesselator() and tesselateInsetContours()).
	stroker->m_NumContourVertices = 0;
	stroker->m_NumContours = 0;
	return true;
}

void strokerConcaveFillAddContour(Stroker* stroker, const float* vertexList, uint32_t numVertices)
{
	// The contours are added to the tesselator only if they have to be tesselated (see
	// strokerConcaveFillEnd()/strokerConcaveFillEndAA()).
	if (stroker->m_NumContourVertices + numVertices > stroker->m_ContourVertexCapacity) {
		const uint32_t newCapacity = bx::max<uint32_t>(stroker->m_NumContourVertices + numVertices, stroker->m_ContourVertexCapacity + (stroker->m_ContourVertexCapacity >> 1));
		stroker->m_ContourVertices = (Vec2*)bx::alignedRealloc(stroker->m_Allocator, stroker->m_ContourVertices, sizeof(Vec2) * newCapacity, 16);
		stroker->m_ContourVertexCapacity = newCapacity;
	}
	if (stroker->m_NumContours + 1 > stroker->m_ContourCapacity) {
		const uint32_t newCapacity = bx::max<uint32_t>(16, stroker->m_ContourCapacity * 2);
		stroker->m_ContourSizes = (uint32_t*)bx::alignedRealloc(stroker->m_Allocator, stroker->m_ContourSizes, sizeof(uint32_t) * newCapacity, 16);
		stroker->m_ContourCapacity = newCapacity;
	}

	bx::memCopy(&stroker->m_ContourVertices[stroker->m_NumContourVertices], vertexList, sizeof(Vec2) * numVertices);
	stroker->m_NumContourVertices += numVertices;
	stroker->m_ContourSizes[stroker->m_NumContours++] = numVertices;
}

// (Re)creates the tesselator and adds the contours added with strokerConcaveFillAddContour().
static void addContoursToTesselator(Stroker* stroker)
{
	resetTesselator(stroker);
	const Vec2* contourVertices = stroker->m_ContourVertices;
	for (uint32_t i = 0; i < stroker->m_NumContours; ++i) {
		tessAddContour(stroker->m_Tesselator, 2, contourVertices, sizeof(Vec2), (int)stroker->m_ContourSizes[i]);
		contourVertices += stroker->m_ContourSizes[i];
	}
}

static inline double orient2d(const Vec2& a, const Vec2& b, const Vec2& c)
{
	return ((double)b.x - (double)a.x) * ((double)c.y - (double)a.y) - ((double)b.y - (double)a.y) * ((double)c.x - (double)a.x);
}

// Returns true if the segments (a, b) and (c, d) intersect or touch (incl. collinear overlaps).
static bool segmentsIntersect(const Vec2& a, const Vec2& b, const Vec2& c, const Vec2& d)
{
	if (bx::max(a.x, b.x) < bx::min(c.x, d.x) || bx::max(c.x, d.x) < bx::min(a.x, b.x)
	||  bx::max(a.y, b.y) < bx::min(c.y, d.y) || bx::max(c.y, d.y) < bx::min(a.y, b.y)) {
		return false;
	}

	const double o1 = orient2d(a, b, c);
	const double o2 = orient2d(a, b, d);
	const double o3 = orient2d(c, d, a);
	const double o4 = orient2d(c, d, b);
	if (((o1 > 0.0 && o2 < 0.0) || (o1 < 0.0 && o2 > 0.0)) && ((o3 > 0.0 && o4 < 0.0) || (o3 < 0.0 && o4 > 0.0))) {
		return true;
	}

	// Touching (the bounding boxes overlap, so a zero orientation means the point is on the other segment)
	return o1 == 0.0 || o2 == 0.0 || o3 == 0.0 || o4 == 0.0;
}

// Fast path for concave fills: if the only contour is a simple polygon (no self intersections, no touching edges,
// no duplicate vertices) with at most kMaxSimplePolygonVertices vertices, it's triangulated by ear clipping, which is
// much faster than libtess2 for small polygons. Both fill rules fill the interior of a simple polygon. The vertices
// are stored in CCW order in m_SimplePolyVertices (so the boundary contour is 0..n-1, like the tesselator's boundary
// contours, with the interior on the left) and the n - 2 CCW triangles in m_SimplePolyTriangles.
// Returns the number of vertices, or 0 if the fast path can't be used (the contours have to be tesselated).
// Returns a 32-bit key with the same order as the float (NaNs excluded).
static inline uint32_t floatSortKey(float f)
{
	const uint32_t u = bx::floatToBits(f);
	return (u & 0x80000000u) != 0 ? ~u : (u | 0x80000000u);
}

// Sorts the n (unique) keys. Uses a natural merge sort for larger arrays (the keys of the edges of a polygon form
// long monotone runs). Returns the sorted keys (either keys or temp).
static uint64_t* sortPolygonKeys(uint64_t* keys, uint64_t* temp, uint32_t n)
{
	if (n <= 32) {
		for (uint32_t i = 1; i < n; ++i) {
			const uint64_t key = keys[i];
			uint32_t k = i;
			for (; k > 0 && keys[k - 1] > key; --k) {
				keys[k] = keys[k - 1];
			}
			keys[k] = key;
		}

		return keys;
	}

	// Find the runs and make them all ascending
	uint16_t runStart[kMaxSimplePolygonVertices + 1];
	uint32_t numRuns = 0;
	for (uint32_t i = 0; i < n; ) {
		uint32_t j = i + 1;
		if (j < n && keys[j] < keys[i]) {
			while (j < n && keys[j] < keys[j - 1]) {
				++j;
			}

			for (uint32_t l = i, r = j - 1; l < r; ++l, --r) {
				const uint64_t tmp = keys[l];
				keys[l] = keys[r];
				keys[r] = tmp;
			}
		} else {
			while (j < n && keys[j] > keys[j - 1]) {
				++j;
			}
		}

		runStart[numRuns++] = (uint16_t)i;
		i = j;
	}
	runStart[numRuns] = (uint16_t)n;

	uint64_t* src = keys;
	uint64_t* dst = temp;
	while (numRuns > 1) {
		uint32_t numMerged = 0;
		for (uint32_t r = 0; r < numRuns; r += 2) {
			const uint32_t begin = runStart[r];
			const uint32_t mid = runStart[bx::min(r + 1, numRuns)];
			const uint32_t end = runStart[bx::min(r + 2, numRuns)];
			uint32_t a = begin;
			uint32_t b = mid;
			uint32_t k = begin;
			while (a < mid && b < end) {
				dst[k++] = src[a] < src[b] ? src[a++] : src[b++];
			}
			while (a < mid) {
				dst[k++] = src[a++];
			}
			while (b < end) {
				dst[k++] = src[b++];
			}

			runStart[numMerged++] = (uint16_t)begin;
		}
		runStart[numMerged] = (uint16_t)n;
		numRuns = numMerged;

		uint64_t* tmp = src;
		src = dst;
		dst = tmp;
	}

	return src;
}

// Returns true if all the edges of the polygon turn CCW around point c and the polygon winds around it exactly once,
// i.e. the polygon is star-shaped wrt c (c is in its kernel), so it's simple.
static bool windsOnceAround(const Vec2* vtx, uint32_t n, const Vec2& c)
{
	uint32_t numCrossings = 0; // Edges crossing the ray from c towards +X (from below)
	for (uint32_t i = 0; i < n; ++i) {
		const Vec2& a = vtx[i];
		const Vec2& b = vtx[i + 1 == n ? 0 : i + 1];
		if (!(orient2d(c, a, b) > 0.0)) {
			return false;
		}

		numCrossings += (a.y < c.y && b.y >= c.y) ? 1 : 0;
	}

	return numCrossings == 1;
}

// Returns true if any 2 non-adjacent edges of the polygon intersect or touch. Uses a sweep along the X or Y axis:
// the edges are sorted by their min coordinate and each edge is tested only against the following edges whose range
// overlaps with its own.
static bool hasIntersectingEdges(const Vec2* vtx, uint32_t n, bool sweepY)
{
	float edgeMin[kMaxSimplePolygonVertices];
	float edgeMax[kMaxSimplePolygonVertices];
	uint64_t sortKeys[kMaxSimplePolygonVertices];
	uint64_t sortTemp[kMaxSimplePolygonVertices];
	for (uint32_t i = 0; i < n; ++i) {
		const Vec2& p1 = vtx[i];
		const Vec2& p2 = vtx[i + 1 == n ? 0 : i + 1];
		const float c1 = sweepY ? p1.y : p1.x;
		const float c2 = sweepY ? p2.y : p2.x;
		edgeMin[i] = bx::min(c1, c2);
		edgeMax[i] = bx::max(c1, c2);
		sortKeys[i] = ((uint64_t)floatSortKey(edgeMin[i]) << 32) | i;
	}

	const uint64_t* sortedEdges = sortPolygonKeys(sortKeys, sortTemp, n);
	for (uint32_t i = 0; i < n; ++i) {
		const uint32_t ea = (uint32_t)sortedEdges[i];
		const float maxC = edgeMax[ea];
		const Vec2& a = vtx[ea];
		const Vec2& b = vtx[ea + 1 == n ? 0 : ea + 1];
		for (uint32_t j = i + 1; j < n; ++j) {
			const uint32_t eb = (uint32_t)sortedEdges[j];
			if (edgeMin[eb] > maxC) {
				break;
			}

			const uint32_t d = ea > eb ? ea - eb : eb - ea;
			if (d == 1 || d == n - 1) {
				continue; // Adjacent
			}

			if (segmentsIntersect(a, b, vtx[eb], vtx[eb + 1 == n ? 0 : eb + 1])) {
				return true;
			}
		}
	}

	return false;
}

// Uniform grid over the bounding box of a polygon (see triangulateSimplePolygon()).
struct PolygonGrid
{
	static const uint32_t kMaxSize = 16;
	static const uint32_t kMaxCells = kMaxSize * kMaxSize;

	Vec2 m_Min;
	Vec2 m_InvCellSize;
	uint32_t m_Size; // Number of cells along each axis

	void init(const Vec2& bbMin, const Vec2& bbMax, uint32_t numElements)
	{
		// ~2 elements per cell (at least 1 cell)
		uint32_t size = 1;
		while (size < kMaxSize && size * size * 2 < numElements) {
			++size;
		}

		m_Size = size;
		m_Min = bbMin;
		m_InvCellSize.x = bbMax.x > bbMin.x ? (float)size / (bbMax.x - bbMin.x) : 0.0f;
		m_InvCellSize.y = bbMax.y > bbMin.y ? (float)size / (bbMax.y - bbMin.y) : 0.0f;
	}

	// NOTE: Monotonic, so the cell range of a bounding box includes the cells of all the points inside it.
	uint32_t getCell(float v, float vmin, float invCellSize) const
	{
		const float c = (v - vmin) * invCellSize;
		return c <= 0.0f ? 0 : bx::min((uint32_t)c, m_Size - 1);
	}

	uint32_t getCellID(const Vec2& p) const
	{
		return getCell(p.y, m_Min.y, m_InvCellSize.y) * m_Size + getCell(p.x, m_Min.x, m_InvCellSize.x);
	}

	// Cell range (minX, minY, maxX, maxY) overlapped by bbox (minX, minY, maxX, maxY)
	void getCellRange(const float* bbox, uint8_t* range) const
	{
		range[0] = (uint8_t)getCell(bbox[0], m_Min.x, m_InvCellSize.x);
		range[1] = (uint8_t)getCell(bbox[1], m_Min.y, m_InvCellSize.y);
		range[2] = (uint8_t)getCell(bbox[2], m_Min.x, m_InvCellSize.x);
		range[3] = (uint8_t)getCell(bbox[3], m_Min.y, m_InvCellSize.y);
	}
};

// Returns true if the reflex vertex ir is inside or on the ear (ip, ic, in) with the bounding box bbox.
static inline bool blocksEar(const Vec2* vtx, uint32_t ir, uint32_t ip, uint32_t ic, uint32_t in, const float* bbox)
{
	const Vec2& r = vtx[ir];
	if (ir == ip || ir == ic || ir == in || r.x < bbox[0] || r.y < bbox[1] || r.x > bbox[2] || r.y > bbox[3]) {
		return false;
	}

	const Vec2& p = vtx[ip];
	const Vec2& c = vtx[ic];
	const Vec2& q = vtx[in];
	return orient2d(p, c, r) >= 0.0 && orient2d(c, q, r) >= 0.0 && orient2d(q, p, r) >= 0.0;
}

static uint32_t triangulateSimplePolygon(Stroker* stroker)
{
	if (stroker->m_NumContours != 1) {
		return 0;
	}

	const uint32_t n = stroker->m_ContourSizes[0];
	if (n < 3 || n > kMaxSimplePolygonVertices) {
		return 0;
	}

	const Vec2* src = stroker->m_ContourVertices;

	// Orientation (and degenerate polygons)
	double area2 = 0.0;
	for (uint32_t i = 0, j = n - 1; i < n; j = i++) {
		area2 += (double)src[j].x * (double)src[i].y - (double)src[i].x * (double)src[j].y;
	}
	if (!(bx::abs(area2) > 1e-6) || area2 != area2) {
		return 0;
	}

	Vec2* vtx = stroker->m_SimplePolyVertices;
	if (area2 > 0.0) {
		bx::memCopy(vtx, src, n * sizeof(Vec2));
	} else {
		for (uint32_t i = 0; i < n; ++i) {
			vtx[i] = src[n - 1 - i];
		}
	}

	// Simplicity: adjacent edges must not fold back onto each other, other edges must not intersect or touch
	// (this also rejects duplicate vertices).
	bool isReflex[kMaxSimplePolygonVertices]; // Reflex or flat vertices (see ear clipping below)
	float sumDx = 0.0f;
	float sumDy = 0.0f;
	double sumX = 0.0;
	double sumY = 0.0;
	Vec2 bbMin = vtx[0];
	Vec2 bbMax = vtx[0];
	for (uint32_t i = 0; i < n; ++i) {
		const Vec2& p0 = vtx[i == 0 ? n - 1 : i - 1];
		const Vec2& p1 = vtx[i];
		const Vec2& p2 = vtx[i + 1 == n ? 0 : i + 1];
		if (p0.x == p1.x && p0.y == p1.y) {
			return 0;
		}
		const double o = orient2d(p0, p1, p2);
		if (o == 0.0 && ((double)p1.x - p0.x) * ((double)p2.x - p1.x) + ((double)p1.y - p0.y) * ((double)p2.y - p1.y) <= 0.0) {
			return 0;
		}

		isReflex[i] = !(o > 0.0);
		sumDx += bx::abs(p2.x - p1.x);
		sumDy += bx::abs(p2.y - p1.y);
		sumX += p1.x;
		sumY += p1.y;
		bbMin = { bx::min(bbMin.x, p1.x), bx::min(bbMin.y, p1.y) };
		bbMax = { bx::max(bbMax.x, p1.x), bx::max(bbMax.y, p1.y) };
	}

	// Star-shaped polygons (e.g. stars, circles, rounded shapes) are simple if they wind exactly once around a point
	// of their kernel (the average of the vertices is tried). Other polygons are tested with a sweep.
	const Vec2 center = { (float)(sumX / n), (float)(sumY / n) };
	if (!windsOnceAround(vtx, n, center)
	&&  hasIntersectingEdges(vtx, n, sumDx * (bbMax.y - bbMin.y) > sumDy * (bbMax.x - bbMin.x))) {
		return 0;
	}


	// Ear clipping. An ear (p, c, q) must be convex (or c on the segment p-q) and no other vertex may be inside
	// or on the triangle. In a simple polygon it's enough to test the reflex vertices (if there's a vertex inside
	// the triangle, there's also a reflex one). Flat vertices are tested as well.
	// The reflex vertices are binned into the grid (a linked list per cell: reflexHead[cell] -> reflexNext[vertex]
	// -> ... -> UINT16_MAX) so each ear is tested only against the reflex vertices in the cells its bounding box
	// overlaps. Vertices which become reflex later (only possible due to numerical issues) are added then.
	uint16_t prev[kMaxSimplePolygonVertices];
	uint16_t next[kMaxSimplePolygonVertices];
	uint16_t reflexHead[PolygonGrid::kMaxCells];
	uint16_t reflexNext[kMaxSimplePolygonVertices];
	bool isListed[kMaxSimplePolygonVertices];
	uint32_t numReflex = 0; // Reflex vertices of the remaining polygon
	for (uint32_t i = 0; i < n; ++i) {
		numReflex += isReflex[i] ? 1 : 0;
	}

	PolygonGrid grid;
	grid.init(bbMin, bbMax, n);
	bx::memSet(reflexHead, 0xff, sizeof(uint16_t) * grid.m_Size * grid.m_Size);
	for (uint32_t i = 0; i < n; ++i) {
		prev[i] = (uint16_t)(i == 0 ? n - 1 : i - 1);
		next[i] = (uint16_t)(i + 1 == n ? 0 : i + 1);
		isListed[i] = isReflex[i];
		if (isReflex[i]) {
			const uint32_t cell = grid.getCellID(vtx[i]);
			reflexNext[i] = reflexHead[cell];
			reflexHead[cell] = (uint16_t)i;
		}
	}

	// Bail out (and let libtess2 handle the polygon) if ear clipping takes too long.
	int32_t budget = (int32_t)(n * 64 + 256);

	uint16_t* tri = stroker->m_SimplePolyTriangles;
	uint32_t numRemaining = n;
	uint32_t c = 0;
	uint32_t numTested = 0;
	while (numRemaining > 3) {
		if (numTested == numRemaining || budget < 0) {
			return 0; // No ear found (numerical problems) or too slow
		}
		--budget;

		const uint32_t ip = prev[c];
		const uint32_t in = next[c];
		const Vec2& pp = vtx[ip];
		const Vec2& pc = vtx[c];
		const Vec2& pq = vtx[in];
		const double o = orient2d(pp, pc, pq);
		bool isEar = o > 0.0 || (o == 0.0 && ((double)pc.x - pp.x) * ((double)pq.x - pc.x) + ((double)pc.y - pp.y) * ((double)pq.y - pc.y) > 0.0);
		if (isEar && numReflex != 0) {
			const float bbox[4] = {
				bx::min(pp.x, bx::min(pc.x, pq.x)),
				bx::min(pp.y, bx::min(pc.y, pq.y)),
				bx::max(pp.x, bx::max(pc.x, pq.x)),
				bx::max(pp.y, bx::max(pc.y, pq.y))
			};
			uint8_t cellRange[4];
			grid.getCellRange(bbox, cellRange);
			for (uint32_t cy = cellRange[1]; cy <= cellRange[3] && isEar; ++cy) {
				for (uint32_t cx = cellRange[0]; cx <= cellRange[2] && isEar; ++cx) {
					--budget;
					uint16_t* link = &reflexHead[cy * grid.m_Size + cx];
					while (*link != UINT16_MAX && isEar) {
						--budget;
						const uint32_t ir = *link;
						if (!isReflex[ir]) {
							// Clipped or not reflex anymore: remove it from the grid
							*link = reflexNext[ir];
							isListed[ir] = false;
							continue;
						}

						isEar = !blocksEar(vtx, ir, ip, c, in, bbox);
						link = &reflexNext[ir];
					}
				}
			}
		}

		if (!isEar) {
			c = in;
			++numTested;
			continue;
		}

		tri[0] = (uint16_t)ip;
		tri[1] = (uint16_t)c;
		tri[2] = (uint16_t)in;
		tri += 3;

		next[ip] = (uint16_t)in;
		prev[in] = (uint16_t)ip;
		numReflex -= isReflex[c] ? 1 : 0;
		isReflex[c] = false;
		--numRemaining;

		// The neighbors' angles changed.
		const uint32_t neighbors[2] = { ip, in };
		for (uint32_t k = 0; k < 2; ++k) {
			const uint32_t iv = neighbors[k];
			const bool wasReflex = isReflex[iv];
			isReflex[iv] = !(orient2d(vtx[prev[iv]], vtx[iv], vtx[next[iv]]) > 0.0);
			numReflex = numReflex + (isReflex[iv] ? 1 : 0) - (wasReflex ? 1 : 0);
			if (isReflex[iv] && !isListed[iv]) {
				const uint32_t cell = grid.getCellID(vtx[iv]);
				reflexNext[iv] = reflexHead[cell];
				reflexHead[cell] = (uint16_t)iv;
				isListed[iv] = true;
			}
		}

		c = in;
		numTested = 0;
	}

	tri[0] = prev[c];
	tri[1] = (uint16_t)c;
	tri[2] = next[c];

	// Boundary: vertices 0..n-1, a single contour (first = 0, count = n)
	uint16_t* boundary = stroker->m_SimplePolyBoundary;
	for (uint32_t i = 0; i < n; ++i) {
		boundary[i] = (uint16_t)i;
	}
	boundary[n + 0] = 0;
	boundary[n + 1] = (uint16_t)n;

	return n;
}

bool strokerConcaveFillEnd(Stroker* stroker, Mesh* mesh, FillRule::Enum fillRule)
{
	const int windingRule = fillRule == vg::FillRule::NonZero ? TESS_WINDING_NONZERO : TESS_WINDING_ODD;

	// All contours are on the XY plane so pass the normal explicitly instead of letting libtess2 compute it.
	// NOTE: With a fixed normal libtess2 skips its orientation check so the output triangles are always CCW
	// in XY space (previously the winding depended on the input). NonZero and EvenOdd are symmetric wrt the
	// sign of the winding number, so the filled region is the same either way (the sweep direction might
	// differ, so the exact triangulation can differ on degenerate input).
	const uint32_t numSimplePolyVertices = triangulateSimplePolygon(stroker);
	if (numSimplePolyVertices != 0) {
		mesh->m_PosBuffer = &stroker->m_SimplePolyVertices[0].x;
		mesh->m_ColorBuffer = nullptr;
		mesh->m_IndexBuffer = stroker->m_SimplePolyTriangles;
		mesh->m_NumVertices = numSimplePolyVertices;
		mesh->m_NumIndices = (numSimplePolyVertices - 2) * 3;
		return true;
	}

	addContoursToTesselator(stroker);

	const float normal[3] = { 0.0f, 0.0f, 1.0f };
	if (!tessTesselate(stroker->m_Tesselator, windingRule, TESS_POLYGONS, 3, 2, &normal[0])) {
		return false;
	}

	mesh->m_PosBuffer = tessGetVertices(stroker->m_Tesselator);
	mesh->m_ColorBuffer = nullptr;
	mesh->m_IndexBuffer = tessGetElements(stroker->m_Tesselator);
	mesh->m_NumVertices = (uint32_t)tessGetVertexCount(stroker->m_Tesselator);
	mesh->m_NumIndices = (uint32_t)tessGetElementCount(stroker->m_Tesselator) * 3;

	return true;
}

// Generates the AA fringes of the boundary contours of a tesselated area (with the interior on their left side):
// 2 vertices per contour vertex (inner, outer) and a quad (6 indices) per contour segment. Contour i consists of
// contours[i * 2 + 1] vertices; its j-th vertex is vertices[indices[contours[i * 2] + j]] (indices can be null for
// sequential contour vertices).
// NOTE: Compared to the previous implementation of strokerConcaveFillEndAA(), the fringe is also generated when the
// first vertex of a contour is collinear with its neighbors (bx::sign(cross) == 0 used to disable AA for the whole
// contour) and the direction of the closing segment is calculated from the original first vertex (instead of the
// already inset one).
static void generateFringes(Vec2* dstPos, Color* dstColor, uint16_t* dstIndex, const Vec2* vertices, const TESSindex* contours, uint32_t numContours, const TESSindex* indices, float fringeWidth, Color color)
{
	// The inner vertex is p + v * aa (the interior is on the left side of the contour).
	const float aa = fringeWidth * 0.5f;
	const Color c0 = colorSetAlpha(color, 0);
	for (uint32_t iContour = 0; iContour < numContours; ++iContour) {
		const uint32_t first = contours[iContour * 2 + 0];
		const uint32_t numContourVertices = contours[iContour * 2 + 1];

		Vec2 d01 = vec2Dir(vertices[indices ? indices[first + numContourVertices - 1] : first + numContourVertices - 1], vertices[indices ? indices[first] : first]);
		for (uint32_t i = 0; i < numContourVertices; ++i) {
			const uint32_t i2 = i + 1 == numContourVertices ? 0 : i + 1;
			const Vec2& p1 = vertices[indices ? indices[first + i] : first + i];
			const Vec2& p2 = vertices[indices ? indices[first + i2] : first + i2];

			const Vec2 d12 = vec2Dir(p1, p2);
			const Vec2 v_aa = vec2Scale(calcExtrusionVector(d01, d12), aa);

			dstPos[0] = vec2Add(p1, v_aa);
			dstPos[1] = vec2Sub(p1, v_aa);
			dstColor[0] = color;
			dstColor[1] = c0;
			dstPos += 2;
			dstColor += 2;

			d01 = d12;
		}

		const uint16_t firstID = (uint16_t)(first * 2);
		const uint32_t numSegments = numContourVertices - 1;
		for (uint32_t iSegment = 0; iSegment < numSegments; ++iSegment) {
			const uint16_t id0 = (uint16_t)(firstID + iSegment * 2);
			dstIndex[0] = id0;
			dstIndex[1] = (uint16_t)(id0 + 2);
			dstIndex[2] = (uint16_t)(id0 + 1);
			dstIndex[3] = (uint16_t)(id0 + 2);
			dstIndex[4] = (uint16_t)(id0 + 3);
			dstIndex[5] = (uint16_t)(id0 + 1);
			dstIndex += 6;
		}

		// Last (closing) segment
		{
			const uint16_t id0 = (uint16_t)(firstID + numSegments * 2);
			dstIndex[0] = id0;
			dstIndex[1] = firstID;
			dstIndex[2] = (uint16_t)(id0 + 1);
			dstIndex[3] = firstID;
			dstIndex[4] = (uint16_t)(firstID + 1);
			dstIndex[5] = (uint16_t)(id0 + 1);
			dstIndex += 6;
		}
	}
}

// Twice the signed area of triangle (i0, i1, i2) (positive if CCW).
static inline float triArea2(const Vec2* pos, uint16_t i0, uint16_t i1, uint16_t i2)
{
	const Vec2 a = pos[i0];
	const Vec2 b = pos[i1];
	const Vec2 c = pos[i2];
	return (b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y);
}

// Directed edge (a -> b) -> triangle hash table used by fixFlippedTriangles() (open addressing, linear probing).
struct EdgeHash
{
	uint32_t* m_Keys;   // (a << 16) | b, UINT32_MAX for empty slots
	uint32_t* m_Values; // Offset of the triangle's first index
	uint32_t m_Mask;
	uint32_t m_Shift;   // 32 - log2(capacity)
};

static inline uint32_t edgeKey(uint16_t a, uint16_t b)
{
	return ((uint32_t)a << 16) | b;
}

static inline uint32_t* edgeHashSlot(EdgeHash* hash, uint32_t key)
{
	// Fibonacci hashing: the high bits of the product depend on all the bits of the key (the low bits only on
	// the low bits of the key, i.e. of the edge's 2nd vertex).
	uint32_t slot = (key * 2654435761u) >> hash->m_Shift;
	while (hash->m_Keys[slot] != key && hash->m_Keys[slot] != UINT32_MAX) {
		slot = (slot + 1) & hash->m_Mask;
	}

	return &hash->m_Keys[slot];
}

static inline void edgeHashSet(EdgeHash* hash, uint16_t a, uint16_t b, uint32_t tri)
{
	const uint32_t key = edgeKey(a, b);
	uint32_t* slot = edgeHashSlot(hash, key);
	*slot = key;
	hash->m_Values[slot - hash->m_Keys] = tri;
}

// Tries to make triangle 'tri' CCW by flipping one of its edges, i.e. replacing it and the neighbor on the other
// side of the edge with the 2 triangles on the other diagonal of the quad they form. The flip is accepted only if
// both new triangles are CCW (see fixFlippedTriangles() for minArea2). Returns false if none of the triangle's edges
// can be flipped.
static bool flipEdgeOfTriangle(const Vec2* pos, uint16_t* tris, EdgeHash* hash, uint32_t tri, float minArea2)
{
	uint16_t* t = &tris[tri];
	for (uint32_t e = 0; e < 3; ++e) {
		const uint16_t a = t[e];
		const uint16_t b = t[e == 2 ? 0 : e + 1];
		const uint16_t c = t[e == 0 ? 2 : e - 1];

		// Find the neighbor (b, a, d). The hash table can contain stale entries (edges removed by previous flips),
		// so check that the triangle still has the edge.
		const uint32_t* slot = edgeHashSlot(hash, edgeKey(b, a));
		if (*slot == UINT32_MAX) {
			continue;
		}

		const uint32_t n = hash->m_Values[slot - hash->m_Keys];
		uint16_t* u = &tris[n];
		const uint32_t k = u[0] == b ? 0 : (u[1] == b ? 1 : (u[2] == b ? 2 : 3));
		if (n == tri || k == 3 || u[k == 2 ? 0 : k + 1] != a) {
			continue;
		}

		const uint16_t d = u[k == 0 ? 2 : k - 1];
		if (triArea2(pos, a, d, c) > minArea2 && triArea2(pos, d, b, c) > minArea2) {
			t[0] = a; t[1] = d; t[2] = c;
			u[0] = d; u[1] = b; u[2] = c;
			edgeHashSet(hash, a, d, tri);
			edgeHashSet(hash, d, c, tri);
			edgeHashSet(hash, c, a, tri);
			edgeHashSet(hash, d, b, n);
			edgeHashSet(hash, b, c, n);
			edgeHashSet(hash, c, d, n);
			return true;
		}
	}

	return false;
}

// Finds the triangle on the other side of edge (a -> b), i.e. the triangle with edge (b -> a). Returns UINT32_MAX if
// there's none. The hash table can contain stale entries (edges removed by previous changes), so the triangle is
// checked.
static uint32_t findNeighbor(const uint16_t* tris, EdgeHash* hash, uint16_t a, uint16_t b)
{
	const uint32_t* slot = edgeHashSlot(hash, edgeKey(b, a));
	if (*slot == UINT32_MAX) {
		return UINT32_MAX;
	}

	const uint32_t n = hash->m_Values[slot - hash->m_Keys];
	const uint16_t* u = &tris[n];
	if ((u[0] == b && u[1] == a) || (u[1] == b && u[2] == a) || (u[2] == b && u[0] == a)) {
		return n;
	}

	return UINT32_MAX;
}

static const uint32_t kMaxCavityTris = 32;

// Retriangulates a cavity (the triangle 'tri' and 2 rings of its neighbors, at most kMaxCavityTris triangles)
// so that all its triangles are CCW. The cavity must be a topological disk; its boundary cycle is retriangulated by
// ear clipping (only CCW ears, not containing other boundary vertices). Any triangulation of the boundary cycle has
// the same signed coverage as the original triangles (as with edge flips). Cavity vertices which aren't on its
// boundary are dropped. The triangulation has fewer triangles than the cavity if there were such vertices; the
// remaining slots get degenerate triangles (removed at the end by fixFlippedTriangles()).
static bool retriangulateCavity(const Vec2* pos, uint16_t* tris, EdgeHash* hash, uint32_t tri, float minArea2)
{
	uint32_t cavity[kMaxCavityTris];
	uint32_t numCavityTris = 1;
	cavity[0] = tri;

	uint32_t ringStart = 0;
	for (uint32_t ring = 0; ring < 2; ++ring) {
		// Add the neighbors of the last ring.
		const uint32_t ringEnd = numCavityTris;
		for (uint32_t c = ringStart; c < ringEnd; ++c) {
			const uint16_t* t = &tris[cavity[c]];
			for (uint32_t e = 0; e < 3; ++e) {
				const uint32_t n = findNeighbor(tris, hash, t[e], t[e == 2 ? 0 : e + 1]);
				if (n == UINT32_MAX || numCavityTris == kMaxCavityTris) {
					continue;
				}

				bool found = false;
				for (uint32_t k = 0; k < numCavityTris && !found; ++k) {
					found = cavity[k] == n;
				}
				if (!found) {
					cavity[numCavityTris++] = n;
				}
			}
		}
		ringStart = ringEnd;
		if (numCavityTris == ringEnd) {
			return false; // No new neighbors
		}

		// The triangle and its direct neighbors can't be retriangulated any better than by edge flips, so only try
		// after adding the second ring.
		if (ring == 0) {
			continue;
		}

		// Boundary edges: edges of the cavity triangles whose twin isn't in the cavity.
		uint16_t edgeFrom[kMaxCavityTris * 3];
		uint16_t edgeTo[kMaxCavityTris * 3];
		uint32_t numEdges = 0;
		for (uint32_t c = 0; c < numCavityTris; ++c) {
			const uint16_t* t = &tris[cavity[c]];
			for (uint32_t e = 0; e < 3; ++e) {
				const uint16_t a = t[e];
				const uint16_t b = t[e == 2 ? 0 : e + 1];
				bool interior = false;
				for (uint32_t k = 0; k < numCavityTris && !interior; ++k) {
					const uint16_t* u = &tris[cavity[k]];
					interior = (u[0] == b && u[1] == a) || (u[1] == b && u[2] == a) || (u[2] == b && u[0] == a);
				}
				if (!interior) {
					edgeFrom[numEdges] = a;
					edgeTo[numEdges] = b;
					++numEdges;
				}
			}
		}

		// The boundary must be a single simple cycle (each vertex is the origin of exactly one boundary edge).
		uint16_t poly[kMaxCavityTris * 3];
		uint32_t numPolyVerts = 0;
		bool isDisk = numEdges >= 3;
		for (uint32_t k = 0; k < numEdges && isDisk; ++k) {
			for (uint32_t l = k + 1; l < numEdges && isDisk; ++l) {
				isDisk = edgeFrom[k] != edgeFrom[l];
			}
		}
		if (isDisk) {
			uint32_t e = 0;
			do {
				poly[numPolyVerts++] = edgeFrom[e];
				uint32_t next = numEdges;
				for (uint32_t k = 0; k < numEdges; ++k) {
					if (edgeFrom[k] == edgeTo[e]) {
						next = k;
						break;
					}
				}
				if (next == numEdges) {
					isDisk = false;
					break;
				}
				e = next;
			} while (e != 0 && numPolyVerts < numEdges);
			isDisk = isDisk && e == 0 && numPolyVerts == numEdges;
		}
		if (!isDisk || numPolyVerts - 2 > numCavityTris) {
			continue;
		}

		// Ear clipping
		uint16_t newTris[kMaxCavityTris * 3];
		uint32_t numNewTris = 0;
		uint16_t verts[kMaxCavityTris * 3];
		uint32_t numVerts = numPolyVerts;
		for (uint32_t k = 0; k < numVerts; ++k) {
			verts[k] = poly[k];
		}
		while (numVerts > 3) {
			uint32_t ear = numVerts;
			for (uint32_t k = 0; k < numVerts && ear == numVerts; ++k) {
				const uint16_t a = verts[k == 0 ? numVerts - 1 : k - 1];
				const uint16_t b = verts[k];
				const uint16_t c = verts[k + 1 == numVerts ? 0 : k + 1];
				if (!(triArea2(pos, a, b, c) > 0.0f)) {
					continue;
				}

				bool empty = true;
				for (uint32_t l = 0; l < numVerts && empty; ++l) {
					const uint16_t v = verts[l];
					if (v == a || v == b || v == c) {
						continue;
					}
					empty = !(triArea2(pos, a, b, v) >= 0.0f && triArea2(pos, b, c, v) >= 0.0f && triArea2(pos, c, a, v) >= 0.0f);
				}
				if (empty) {
					ear = k;
				}
			}
			if (ear == numVerts) {
				break;
			}

			newTris[numNewTris * 3 + 0] = verts[ear == 0 ? numVerts - 1 : ear - 1];
			newTris[numNewTris * 3 + 1] = verts[ear];
			newTris[numNewTris * 3 + 2] = verts[ear + 1 == numVerts ? 0 : ear + 1];
			++numNewTris;
			for (uint32_t k = ear; k + 1 < numVerts; ++k) {
				verts[k] = verts[k + 1];
			}
			--numVerts;
		}
		if (numVerts > 3 || !(triArea2(pos, verts[0], verts[1], verts[2]) > minArea2)) {
			continue;
		}
		newTris[numNewTris * 3 + 0] = verts[0];
		newTris[numNewTris * 3 + 1] = verts[1];
		newTris[numNewTris * 3 + 2] = verts[2];
		++numNewTris;

		// Replace the cavity triangles.
		for (uint32_t c = 0; c < numCavityTris; ++c) {
			uint16_t* t = &tris[cavity[c]];
			if (c < numNewTris) {
				t[0] = newTris[c * 3 + 0];
				t[1] = newTris[c * 3 + 1];
				t[2] = newTris[c * 3 + 2];
				edgeHashSet(hash, t[0], t[1], cavity[c]);
				edgeHashSet(hash, t[1], t[2], cavity[c]);
				edgeHashSet(hash, t[2], t[0], cavity[c]);
			} else {
				t[0] = t[1] = t[2] = newTris[0]; // Degenerate
			}
		}

		return true;
	}

	return false;
}

// Makes all triangles CCW by flipping edges and, if that isn't enough, by retriangulating small cavities around the
// flipped triangles. Both keep the (signed) coverage of the triangles, so if all triangles end up CCW they cover the
// same area as the tesselation of the inset contours would (when the inset contours don't intersect themselves;
// otherwise the triangles can't all be CCW). The fast path (no flipped triangles) is a single pass over the
// triangles; the edge hash table is built only if needed. Returns false if some triangles couldn't be fixed.
// '*numIndices' is updated (degenerate triangles left by retriangulateCavity() are removed).
// Degenerate (almost collinear) triangles with a tiny negative area are accepted (they don't cover anything visible
// and usually can't be fixed, e.g. 3 consecutive inset vertices on an almost straight part of a contour).
static bool fixFlippedTriangles(Stroker* stroker, const Vec2* pos, uint16_t* tris, uint32_t* numIndicesInOut)
{
	const uint32_t numIndices = *numIndicesInOut;
	const float minArea2 = -stroker->m_FringeWidth * stroker->m_FringeWidth * 1e-3f;

	uint32_t i = 0;
	while (i < numIndices && triArea2(pos, tris[i], tris[i + 1], tris[i + 2]) > minArea2) {
		i += 3;
	}

	if (i == numIndices) {
		return true;
	}

	// Build the edge hash table (load factor <= 0.5)
	uint32_t capacity = 16;
	uint32_t capacityBits = 4;
	while (capacity < numIndices * 2) {
		capacity <<= 1;
		++capacityBits;
	}

	if (capacity > stroker->m_EdgeHashCapacity) {
		if (stroker->m_EdgeHash) {
			bx::alignedFree(stroker->m_Allocator, stroker->m_EdgeHash, 16);
		}
		stroker->m_EdgeHash = (uint32_t*)bx::alignedAlloc(stroker->m_Allocator, sizeof(uint32_t) * 2 * capacity, 16);
		stroker->m_EdgeHashCapacity = capacity;
	}

	EdgeHash hash;
	hash.m_Keys = stroker->m_EdgeHash;
	hash.m_Values = stroker->m_EdgeHash + capacity;
	hash.m_Mask = capacity - 1;
	hash.m_Shift = 32 - capacityBits;
	bx::memSet(hash.m_Keys, 0xFF, sizeof(uint32_t) * capacity);
	for (uint32_t t = 0; t < numIndices; t += 3) {
		edgeHashSet(&hash, tris[t + 0], tris[t + 1], t);
		edgeHashSet(&hash, tris[t + 1], tris[t + 2], t);
		edgeHashSet(&hash, tris[t + 2], tris[t + 0], t);
	}

	const uint32_t kMaxPasses = 8;
	bool allCCW = false;
	bool retriangulated = false;
	for (uint32_t pass = 0; pass < kMaxPasses && !allCCW; ++pass) {
		allCCW = true;
		bool fixed = false;
		for (; i < numIndices; i += 3) {
			if (!(triArea2(pos, tris[i], tris[i + 1], tris[i + 2]) > minArea2)) {
				if (flipEdgeOfTriangle(pos, tris, &hash, i, minArea2)) {
					fixed = true;
				} else if (retriangulateCavity(pos, tris, &hash, i, minArea2)) {
					fixed = true;
					retriangulated = true;
				} else {
					allCCW = false;
				}
			}
		}

		if (!allCCW && !fixed) {
			break;
		}

		// Fixing a triangle can't flip other triangles, but check everything again to be sure.
		if (allCCW) {
			for (uint32_t t = 0; t < numIndices && allCCW; t += 3) {
				allCCW = triArea2(pos, tris[t], tris[t + 1], tris[t + 2]) > minArea2;
			}
		}

		i = 0;
	}

	if (!retriangulated) {
		return allCCW;
	}

	// Remove the degenerate triangles left by retriangulateCavity().
	uint32_t numOut = 0;
	for (uint32_t t = 0; t < numIndices; t += 3) {
		if (tris[t] == tris[t + 1] && tris[t + 1] == tris[t + 2]) {
			continue;
		}
		tris[numOut + 0] = tris[t + 0];
		tris[numOut + 1] = tris[t + 1];
		tris[numOut + 2] = tris[t + 2];
		numOut += 3;
	}

	*numIndicesInOut = numOut;
	return allCCW;
}

// Moves the fringe vertices of boundary vertex occurrence k by (newScale - oldScale) * v, where v is the inset vector
// (half the fringe, i.e. (inner - outer) / 2).
static void moveFringe(Vec2* pos, uint32_t k, float oldScale, float newScale)
{
	const Vec2 v = vec2Scale(vec2Sub(pos[k * 2 + 0], pos[k * 2 + 1]), 0.5f);
	const Vec2 delta = vec2Scale(v, newScale - oldScale);
	pos[k * 2 + 0] = vec2Add(pos[k * 2 + 0], delta);
	pos[k * 2 + 1] = vec2Add(pos[k * 2 + 1], delta);
}

// Last resort for triangles which can't be fixed by fixFlippedTriangles() (e.g. dense, almost straight parts of a
// contour whose triangles are much thinner than the fringe, or spikes thinner than the fringe): the inset of the
// boundary vertices of the flipped triangles is reduced (1 -> 1/2 -> 1/4 -> 0) until no triangle is flipped. The
// fringe of such a vertex is moved outwards by the same amount (i.e. it keeps its width), so the AA edge is up to half
// the fringe width further out than the exact inset contour. With no inset the triangles are the tesselator's
// triangles which are valid, so this always succeeds unless a tesselator triangle is flipped.
// CCW triangles cover the inset contours exactly once only if they don't fold over each other. Reducing the inset of
// some vertices but not their neighbors can create such folds (e.g. around contour pinches or duplicate vertices), so
// the angles of the triangles around each vertex must not add up to more than a full turn.
// Returns false (and restores the original inset, so that the inset contours can be tesselated instead) on failure.
// Reducing the inset of a vertex can flip one of its other triangles, so the triangles of the changed vertices are
// checked again (worklist). Each vertex changes at most 3 times, so this is O(n).
// 'pos' contains the fringe vertices (inner, outer) of each boundary vertex occurrence ('numFringeVertices').
static bool clampInset(Stroker* stroker, Vec2* pos, const uint16_t* tris, uint32_t numIndices, uint32_t numFringeVertices, uint32_t numVertices)
{
	const float minArea2 = -stroker->m_FringeWidth * stroker->m_FringeWidth * 1e-3f;
	const uint32_t numOccurrences = numFringeVertices / 2;
	const uint32_t numTris = numIndices / 3;

	// Scratch memory: scale (float), original fringe vertices (2 Vec2, restored exactly on failure) and changed
	// occurrences list per occurrence; first incident triangle and a mark per vertex; incident triangles (numIndices);
	// worklist and in-worklist flags per triangle.
	const uint32_t numScratch = numOccurrences * 6 + numVertices * 2 + 1 + numIndices + numTris * 2;
	if (numScratch > stroker->m_ClampScratchCapacity) {
		if (stroker->m_ClampScratch) {
			bx::alignedFree(stroker->m_Allocator, stroker->m_ClampScratch, 16);
		}
		stroker->m_ClampScratch = (uint32_t*)bx::alignedAlloc(stroker->m_Allocator, sizeof(uint32_t) * numScratch, 16);
		stroker->m_ClampScratchCapacity = numScratch;
	}

	float* scale = (float*)stroker->m_ClampScratch;
	Vec2* original = (Vec2*)(scale + numOccurrences);
	uint32_t* changedList = (uint32_t*)(original + numOccurrences * 2);
	uint32_t* first = changedList + numOccurrences;
	uint32_t* mark = first + numVertices + 1;
	uint32_t* incident = mark + numVertices;
	uint32_t* worklist = incident + numIndices;
	uint32_t* inWorklist = worklist + numTris;

	// Incident triangles of each vertex.
	bx::memSet(first, 0, sizeof(uint32_t) * (numVertices + 1));
	for (uint32_t i = 0; i < numIndices; ++i) {
		++first[tris[i] + 1];
	}
	for (uint32_t v = 0; v < numVertices; ++v) {
		first[v + 1] += first[v];
	}
	for (uint32_t i = 0; i < numIndices; ++i) {
		incident[first[tris[i]]++] = i / 3;
	}
	for (uint32_t v = numVertices; v > 0; --v) {
		first[v] = first[v - 1];
	}
	first[0] = 0;

	uint32_t numWork = 0;
	uint32_t numChanged = 0;
	for (uint32_t k = 0; k < numOccurrences; ++k) {
		scale[k] = 1.0f;
	}
	for (uint32_t t = 0; t < numTris; ++t) {
		const bool flipped = !(triArea2(pos, tris[t * 3 + 0], tris[t * 3 + 1], tris[t * 3 + 2]) > minArea2);
		inWorklist[t] = flipped ? 1 : 0;
		if (flipped) {
			worklist[numWork++] = t;
		}
	}

	while (numWork != 0) {
		const uint32_t t = worklist[--numWork];
		inWorklist[t] = 0;

		const uint16_t* tri = &tris[t * 3];
		if (triArea2(pos, tri[0], tri[1], tri[2]) > minArea2) {
			continue;
		}

		bool changed = false;
		for (uint32_t j = 0; j < 3; ++j) {
			const uint32_t id = tri[j];
			if (id >= numFringeVertices) {
				continue; // Interior vertex
			}

			const uint32_t k = id >> 1;
			const float s = scale[k];
			if (s == 0.0f) {
				continue;
			}

			if (s == 1.0f) {
				original[k * 2 + 0] = pos[k * 2 + 0];
				original[k * 2 + 1] = pos[k * 2 + 1];
				changedList[numChanged++] = k;
			}

			// inner = p + s * v, outer = inner - 2 * v (v is the inset vector, i.e. half the fringe).
			const float newScale = s > 0.25f ? s * 0.5f : 0.0f;
			moveFringe(pos, k, s, newScale);
			scale[k] = newScale;
			changed = true;

			for (uint32_t n = first[id]; n < first[id + 1]; ++n) {
				const uint32_t u = incident[n];
				if (!inWorklist[u]) {
					inWorklist[u] = 1;
					worklist[numWork++] = u;
				}
			}
		}

		if (!changed) {
			break; // A tesselator triangle is flipped.
		}
	}

	// Check for folds (see above). Only the triangles of the moved vertices changed, so only the vertices of those
	// triangles have to be checked.
	bool valid = numWork == 0;
	if (valid) {
		bx::memSet(mark, 0, sizeof(uint32_t) * numVertices);
		for (uint32_t c = 0; c < numChanged && valid; ++c) {
			const uint32_t movedID = changedList[c] * 2;
			for (uint32_t n = first[movedID]; n < first[movedID + 1] && valid; ++n) {
				const uint16_t* tri = &tris[incident[n] * 3];
				for (uint32_t j = 0; j < 3 && valid; ++j) {
					const uint32_t id = tri[j];
					if (mark[id]) {
						continue;
					}
					mark[id] = 1;

					float angleSum = 0.0f;
					for (uint32_t m = first[id]; m < first[id + 1]; ++m) {
						const uint16_t* t = &tris[incident[m] * 3];
						const uint32_t corner = t[0] == id ? 0 : (t[1] == id ? 1 : 2);
						const Vec2 a = pos[id];
						const Vec2 ab = vec2Sub(pos[t[(corner + 1) % 3]], a);
						const Vec2 ac = vec2Sub(pos[t[(corner + 2) % 3]], a);
						angleSum += bx::atan2(ab.x * ac.y - ab.y * ac.x, ab.x * ac.x + ab.y * ac.y);
					}
					valid = angleSum < bx::kPi2 + 1e-3f;
				}
			}
		}
	}

	if (!valid) {
		// Restore the original inset.
		for (uint32_t c = 0; c < numChanged; ++c) {
			const uint32_t k = changedList[c];
			pos[k * 2 + 0] = original[k * 2 + 0];
			pos[k * 2 + 1] = original[k * 2 + 1];
		}
	}

	return valid;
}

// Tesselates the inset boundary contours (the inner fringe vertices generated by generateFringes() into the stroker's
// buffers, 'numFringeVertices' vertices and 'numFringeIndices' indices) and appends the triangles to the stroker's
// buffers. NOTE: Invalidates the current output of the tesselator.
static bool tesselateInsetContours(Stroker* stroker, const TESSindex* contours, uint32_t numContours, uint32_t numFringeVertices, uint32_t numFringeIndices, int windingRule, Color color)
{
	resetTesselator(stroker);
	TESStesselator* tess = stroker->m_Tesselator;
	for (uint32_t iContour = 0; iContour < numContours; ++iContour) {
		const uint32_t first = contours[iContour * 2 + 0];
		const uint32_t numContourVertices = contours[iContour * 2 + 1];
		tessAddContour(tess, 2, &stroker->m_PosBuffer[first * 2], sizeof(Vec2) * 2, (int)numContourVertices);
	}

	const float normal[3] = { 0.0f, 0.0f, 1.0f };
	if (!tessTesselate(tess, windingRule, TESS_POLYGONS, 3, 2, &normal[0])) {
		return false;
	}

	stroker->m_NumVertices = numFringeVertices;
	stroker->m_NumIndices = numFringeIndices;

	const uint32_t numInsetVertices = (uint32_t)tessGetVertexCount(tess);
	expandVB(stroker, numInsetVertices);
	bx::memCopy(&stroker->m_PosBuffer[numFringeVertices], tessGetVertices(tess), sizeof(Vec2) * numInsetVertices);
	vgutil::memset32(&stroker->m_ColorBuffer[numFringeVertices], numInsetVertices, &color);
	stroker->m_NumVertices += numInsetVertices;

	const uint32_t numInsetIndices = (uint32_t)tessGetElementCount(tess) * 3;
	expandIB(stroker, numInsetIndices);
	vgutil::batchTransformDrawIndices(tessGetElements(tess), numInsetIndices, &stroker->m_IndexBuffer[numFringeIndices], (uint16_t)numFringeVertices);
	stroker->m_NumIndices += numInsetIndices;

	return true;
}

// Returns a copy of the tesselator's boundary contour ranges (needed by tesselateInsetContours() after the tesselator's
// output has been freed). The copy is stored in the stroker's vertex map buffer.
static const TESSindex* copyContours(Stroker* stroker, const TESSindex* contours, uint32_t numContours)
{
	if (numContours > stroker->m_VertexMapCapacity) {
		const uint32_t newCapacity = bx::max<uint32_t>(numContours, stroker->m_VertexMapCapacity + (stroker->m_VertexMapCapacity >> 1));
		if (stroker->m_VertexMap) {
			bx::alignedFree(stroker->m_Allocator, stroker->m_VertexMap, 16);
		}
		stroker->m_VertexMap = (uint32_t*)bx::alignedAlloc(stroker->m_Allocator, sizeof(uint32_t) * newCapacity, 16);
		stroker->m_VertexMapCapacity = newCapacity;
	}

	TESSindex* copy = (TESSindex*)stroker->m_VertexMap;
	bx::memCopy(copy, contours, sizeof(TESSindex) * 2 * numContours);
	return copy;
}

static void setMeshFromStrokerBuffers(Stroker* stroker, Mesh* mesh)
{
	mesh->m_PosBuffer = &stroker->m_PosBuffer[0].x;
	mesh->m_ColorBuffer = stroker->m_ColorBuffer;
	mesh->m_IndexBuffer = stroker->m_IndexBuffer;
	mesh->m_NumVertices = stroker->m_NumVertices;
	mesh->m_NumIndices = stroker->m_NumIndices;
}

// The original 2 sweep algorithm: tesselate the boundary contours, generate the fringes and tesselate the inset
// contours. Used when the single sweep version fails (e.g. it requires more memory).
static bool concaveFillEndAATwoSweeps(Stroker* stroker, Mesh* mesh, uint32_t color, int windingRule)
{
	// The tesselator's mesh has been consumed. Tesselate the contours again.
	addContoursToTesselator(stroker);
	TESStesselator* tess = stroker->m_Tesselator;

	const float normal[3] = { 0.0f, 0.0f, 1.0f };
	if (!tessTesselate(tess, windingRule, TESS_BOUNDARY_CONTOURS, 1, 2, &normal[0])) {
		return false;
	}

	const uint32_t numContours = (uint32_t)tessGetElementCount(tess);
	if (numContours == 0) {
		return false;
	}

	const uint32_t numBoundaryVertices = (uint32_t)tessGetVertexCount(tess);
	const uint32_t numFringeVertices = numBoundaryVertices * 2;
	const uint32_t numFringeIndices = numBoundaryVertices * 6;

	resetGeometry(stroker);
	expandVB(stroker, numFringeVertices);
	expandIB(stroker, numFringeIndices);
	const TESSindex* contours = tessGetElements(tess);
	generateFringes(stroker->m_PosBuffer, stroker->m_ColorBuffer, stroker->m_IndexBuffer, (const Vec2*)tessGetVertices(tess), contours, numContours, nullptr, stroker->m_FringeWidth, color);

	if (!tesselateInsetContours(stroker, copyContours(stroker, contours, numContours), numContours, numFringeVertices, numFringeIndices, windingRule, color)) {
		return false;
	}

	setMeshFromStrokerBuffers(stroker, mesh);
	return true;
}

bool strokerConcaveFillEndAA(Stroker* stroker, Mesh* mesh, uint32_t color, FillRule::Enum fillRule)
{
	const int windingRule = fillRule == vg::FillRule::NonZero ? TESS_WINDING_NONZERO : TESS_WINDING_ODD;

	// Tesselate the interior and get its boundary contours from the same sweep. The AA fringe is generated
	// around the boundary contours ([-fringeWidth/2, +fringeWidth/2] around each contour) and the interior
	// triangles use the inner fringe vertices instead of the boundary vertices, i.e. the interior is inset by
	// half the fringe width (the same area the tesselation of the inset boundary contours covers).
	// Simple polygons are triangulated by ear clipping (see triangulateSimplePolygon()); the result has the same
	// form as the tesselator's output: the boundary contour is the polygon itself and each triangle corner is a
	// boundary vertex (corner = vertex ID).
	uint32_t numContours, numTessVertices, numTriangleIndices, numBoundaryVertices;
	const Vec2* tessVertices;
	const TESSindex* triangles;
	const TESSindex* corners;
	const TESSindex* contours;
	const TESSindex* boundaryVertices;
	const uint32_t numSimplePolyVertices = triangulateSimplePolygon(stroker);
	if (numSimplePolyVertices != 0) {
		numContours = 1;
		tessVertices = stroker->m_SimplePolyVertices;
		numTessVertices = numSimplePolyVertices;
		triangles = stroker->m_SimplePolyTriangles;
		numTriangleIndices = (numSimplePolyVertices - 2) * 3;
		corners = stroker->m_SimplePolyTriangles;
		boundaryVertices = stroker->m_SimplePolyBoundary;
		contours = &stroker->m_SimplePolyBoundary[numSimplePolyVertices];
		numBoundaryVertices = numSimplePolyVertices;
	} else {
		addContoursToTesselator(stroker);

		const float normal[3] = { 0.0f, 0.0f, 1.0f };
		TESStesselator* tess = stroker->m_Tesselator;
		if (!tessTesselate(tess, windingRule, TESS_POLYGONS_AND_BOUNDARY, 3, 2, &normal[0])) {
			// Triangulating the interior requires more memory than extracting the boundary contours. Try the 2
			// sweep version.
			return concaveFillEndAATwoSweeps(stroker, mesh, color, windingRule);
		}

		numContours = (uint32_t)tessGetBoundaryContourCount(tess);
		if (numContours == 0) {
			return false;
		}

		tessVertices = (const Vec2*)tessGetVertices(tess);
		numTessVertices = (uint32_t)tessGetVertexCount(tess);
		triangles = tessGetElements(tess);
		numTriangleIndices = (uint32_t)tessGetElementCount(tess) * 3;
		corners = tessGetElementCorners(tess);
		contours = tessGetBoundaryContours(tess);
		boundaryVertices = tessGetBoundaryVertices(tess);
		numBoundaryVertices = (uint32_t)tessGetBoundaryVertexCount(tess);
	}

	// Output vertices: 2 fringe vertices (inner, outer) for each boundary vertex occurrence (in contour order),
	// followed by the interior vertices (tesselator vertices which aren't on the boundary).
	if (numTessVertices > stroker->m_VertexMapCapacity) {
		const uint32_t newCapacity = bx::max<uint32_t>(numTessVertices, stroker->m_VertexMapCapacity + (stroker->m_VertexMapCapacity >> 1));
		if (stroker->m_VertexMap) {
			bx::alignedFree(stroker->m_Allocator, stroker->m_VertexMap, 16);
		}
		stroker->m_VertexMap = (uint32_t*)bx::alignedAlloc(stroker->m_Allocator, sizeof(uint32_t) * newCapacity, 16);
		stroker->m_VertexMapCapacity = newCapacity;
	}

	static const uint32_t kUnused = UINT32_MAX;
	static const uint32_t kBoundary = UINT32_MAX - 1;
	uint32_t* vertexMap = stroker->m_VertexMap;
	for (uint32_t i = 0; i < numTessVertices; ++i) {
		vertexMap[i] = kUnused;
	}
	for (uint32_t i = 0; i < numBoundaryVertices; ++i) {
		vertexMap[boundaryVertices[i]] = kBoundary;
	}

	const uint32_t numFringeVertices = numBoundaryVertices * 2;
	const uint32_t numFringeIndices = numBoundaryVertices * 6;
	uint32_t numVertices = numFringeVertices;
	for (uint32_t i = 0; i < numTriangleIndices; ++i) {
		if (corners[i] == TESS_UNDEF) {
			// Not on the boundary (or, which shouldn't happen, a boundary vertex without an occurrence; keep it as is).
			uint32_t* id = &vertexMap[triangles[i]];
			if (*id >= kBoundary) {
				*id = numVertices++;
			}
		}
	}

	resetGeometry(stroker);
	expandVB(stroker, numVertices);
	expandIB(stroker, numFringeIndices + numTriangleIndices);
	generateFringes(stroker->m_PosBuffer, stroker->m_ColorBuffer, stroker->m_IndexBuffer, tessVertices, contours, numContours, boundaryVertices, stroker->m_FringeWidth, color);

	// Interior vertices
	Vec2* dstPos = &stroker->m_PosBuffer[numFringeVertices];
	for (uint32_t i = 0; i < numTessVertices; ++i) {
		const uint32_t id = vertexMap[i];
		if (id < kBoundary) {
			dstPos[id - numFringeVertices] = tessVertices[i];
		}
	}
	vgutil::memset32(&stroker->m_ColorBuffer[numFringeVertices], numVertices - numFringeVertices, &color);

	// Interior triangles
	uint16_t* dstIndex = &stroker->m_IndexBuffer[numFringeIndices];
	for (uint32_t i = 0; i < numTriangleIndices; ++i) {
		const TESSindex corner = corners[i];
		dstIndex[i] = (uint16_t)(corner != TESS_UNDEF ? corner * 2 : vertexMap[triangles[i]]);
	}

	// The interior triangles are valid only if moving their boundary vertices to the inner fringe vertices doesn't
	// flip any of them. This isn't the case if the fringe is wider than the local feature size (e.g. thin spikes,
	// where the inset contours intersect) or with long skinny triangles (common on finely subdivided curves). The
	// latter can usually be fixed locally (edge flips, retriangulating small cavities). The rest is fixed by reducing
	// the inset of the vertices of the flipped triangles (see clampInset()). Only if that fails too (flipped
	// tesselator triangles), tesselate the inset contours instead (a second sweep).
	uint32_t numFixedTriangleIndices = numTriangleIndices;
	if (!fixFlippedTriangles(stroker, stroker->m_PosBuffer, dstIndex, &numFixedTriangleIndices)
	&&  !clampInset(stroker, stroker->m_PosBuffer, dstIndex, numFixedTriangleIndices, numFringeVertices, numVertices)) {
		if (!tesselateInsetContours(stroker, copyContours(stroker, contours, numContours), numContours, numFringeVertices, numFringeIndices, windingRule, color)) {
			return false;
		}

		setMeshFromStrokerBuffers(stroker, mesh);
		return true;
	}

	stroker->m_NumVertices = numVertices;
	stroker->m_NumIndices = numFringeIndices + numFixedTriangleIndices;
	setMeshFromStrokerBuffers(stroker, mesh);

	return true;
}

//////////////////////////////////////////////////////////////////////////
// Templates
//
template<bool _Closed, LineCap::Enum _LineCap, LineJoin::Enum _LineJoin>
void polylineStroke(Stroker* stroker, Mesh* mesh, const Vec2* vtx, uint32_t numPathVertices, float strokeWidth, const StrokerSink* sink)
{
	const uint32_t numSegments = numPathVertices - (_Closed ? 0 : 1);
	const float hsw = strokeWidth * 0.5f;
	const float da = bx::acos((stroker->m_Scale * hsw) / ((stroker->m_Scale * hsw) + stroker->m_TesselationTolerance)) * 2.0f;
	const uint32_t numPointsHalfCircle = bx::max(2u, (uint32_t)bx::ceil(bx::kPi / da));

	// Round caps are generated by rotating the cap direction by a constant angle.
	float cosCapDa = 1.0f, sinCapDa = 0.0f;
	if (!_Closed && _LineCap == LineCap::Round) {
		const float capDa = bx::kPi / (float)(numPointsHalfCircle - 1);
		cosCapDa = bx::cos(capDa);
		sinCapDa = bx::sin(capDa);
	}

	// Precalculate all segment directions and join extrusion vectors.
	const Vec2* segmentDirs;
	const Vec2* extrusionVecs;
	const uint8_t* innerBevelFlags;
	const uint32_t numInnerBevelJoins = calcSegmentDirsAndExtrusions(stroker, vtx, numPathVertices, _Closed, hsw, &segmentDirs, &extrusionVecs, &innerBevelFlags);

	const uint32_t firstSegmentID = _Closed ? 0 : 1;
	const uint32_t numJoins = numSegments > firstSegmentID ? numSegments - firstSegmentID : 0;

	// Precalculate the arcs of round joins.
	const RoundJoinArc* roundJoinArcs = nullptr;
	uint32_t totalArcPoints = 0;
	if (_LineJoin == LineJoin::Round) {
		roundJoinArcs = calcRoundJoinArcs(stroker, segmentDirs, extrusionVecs, _Closed ? numPathVertices : numPathVertices - 1, firstSegmentID, numSegments, hsw, da, &totalArcPoints);
	}

	// Calculate the exact amount of geometry and reserve space for it.
	// Joins: miter: 2 vertices + 6 indices, bevel: 3 vertices + 6 + 3 indices, round: numArcPoints + 2 vertices + 6 + numArcPoints * 3 indices.
	// The 6 indices connect each join to the previous segment. The first join of a closed path is connected by the
	// closing quad instead, so closed paths need no extra indices. Open paths need space for the 2 caps.
	uint32_t numVertices = 0;
	uint32_t numIndices = 0;
	if (_LineJoin == LineJoin::Miter) {
		numVertices = numJoins * 2;
		numIndices = numJoins * 6;
	} else if (_LineJoin == LineJoin::Bevel) {
		numVertices = numJoins * 3;
		numIndices = numJoins * 9;
	} else {
		numVertices = numJoins * 2 + totalArcPoints;
		numIndices = numJoins * 6 + totalArcPoints * 3;
	}

	// Inner bevel joins: miter +1 vertex, bevel/round +2 vertices, +3 indices (see needsInnerBevel()).
	{
		const uint32_t numFanJoins = countInnerBevelFanJoins(innerBevelFlags, numInnerBevelJoins, firstSegmentID, numSegments, _LineJoin, roundJoinArcs);
		numVertices += numInnerBevelJoins + numFanJoins;
		numIndices += numInnerBevelJoins * 3;
	}

	if (!_Closed) {
		if (_LineCap == LineCap::Round) {
			numVertices += numPointsHalfCircle * 2;
			numIndices += (numPointsHalfCircle - 2) * 6 + 6;
		} else {
			numVertices += 4;
			numIndices += 6;
		}
	}

	GeometryOutput out;
	beginGeometry(stroker, sink, numVertices, numIndices, false, &out);

	const Vec2* posStart = out.m_Pos;
	Vec2* dstPos = out.m_Pos;
	uint16_t* dstIndex = out.m_Index;

	Vec2 d01;
	uint16_t prevSegmentLeftID = 0xFFFF;
	uint16_t prevSegmentRightID = 0xFFFF;
	uint16_t firstSegmentLeftID = 0xFFFF;
	uint16_t firstSegmentRightID = 0xFFFF;
	if (!_Closed) {
		// First segment of an open path
		const Vec2& p0 = vtx[0];

		d01 = segmentDirs[0];

		const Vec2 l01 = vec2PerpCCW(d01);

		if (_LineCap == LineCap::Butt) {
			const Vec2 l01_hsw = vec2Scale(l01, hsw);

			dstPos[0] = vec2Add(p0, l01_hsw);
			dstPos[1] = vec2Sub(p0, l01_hsw);
			dstPos += 2;

			prevSegmentLeftID = 0;
			prevSegmentRightID = 1;
		} else if (_LineCap == LineCap::Square) {
			const Vec2 l01_hsw = vec2Scale(l01, hsw);
			const Vec2 d01_hsw = vec2Scale(d01, hsw);

			dstPos[0] = vec2Add(p0, vec2Sub(l01_hsw, d01_hsw));
			dstPos[1] = vec2Sub(p0, vec2Add(l01_hsw, d01_hsw));
			dstPos += 2;

			prevSegmentLeftID = 0;
			prevSegmentRightID = 1;
		} else if (_LineCap == LineCap::Round) {
			Vec2 capDir = vec2ArcDir(l01);
			for (uint32_t i = 0; i < numPointsHalfCircle; ++i) {
				*dstPos++ = { p0.x + capDir.x * hsw, p0.y + capDir.y * hsw };

				capDir = vec2Rotate(capDir, cosCapDa, sinCapDa);
			}

			for (uint32_t i = 0; i < numPointsHalfCircle - 2; ++i) {
				dstIndex[0] = 0;
				dstIndex[1] = (uint16_t)(i + 1);
				dstIndex[2] = (uint16_t)(i + 2);
				dstIndex += 3;
			}

			prevSegmentLeftID = 0;
			prevSegmentRightID = (uint16_t)(numPointsHalfCircle - 1);
		} else {
			VG_CHECK(false, "Unknown line cap type");
		}
	} else {
		d01 = segmentDirs[numPathVertices - 1];
	}

	for (uint32_t iSegment = firstSegmentID; iSegment < numSegments; ++iSegment) {
		const Vec2& p1 = vtx[iSegment];
		const Vec2 d12 = segmentDirs[iSegment];
		const Vec2 v = extrusionVecs[iSegment];
		const Vec2 v_hsw = vec2Scale(v, hsw);

		// Check which one of the points is the inner corner.
		float leftPointProjDist = d12.x * v_hsw.x + d12.y * v_hsw.y;
		if (innerBevelFlags[iSegment]) {
			// Inner bevel: [X0, X1] (inner corners of the 2 segments), then the outer side: [miter point] or
			// [join point, arc/bevel points...]. X0-M-X1 (miter) or the fan around the join point and X0-P-X1 cover
			// the join.
			const bool leftInner = leftPointProjDist >= 0.0f;
			const float innerHsw = leftInner ? hsw : -hsw;
			const Vec2 l01 = vec2PerpCCW(d01);
			const Vec2 l12 = vec2PerpCCW(d12);
			const uint16_t firstVertexID = (uint16_t)(dstPos - posStart);
			const uint16_t x0 = firstVertexID;
			const uint16_t x1 = (uint16_t)(firstVertexID + 1);
			dstPos[0] = vec2Add(p1, vec2Scale(l01, innerHsw));
			dstPos[1] = vec2Add(p1, vec2Scale(l12, innerHsw));
			dstPos += 2;

			uint16_t outerFirst, outerLast;
			if (_LineJoin == LineJoin::Miter || (_LineJoin == LineJoin::Round && roundJoinArcs[iSegment].m_NumArcPoints == 0)) {
				*dstPos++ = leftInner ? vec2Sub(p1, v_hsw) : vec2Add(p1, v_hsw);
				outerFirst = outerLast = (uint16_t)(firstVertexID + 2);

				const uint16_t id[3] = { x0, outerFirst, x1 };
				dstIndex = copyIndices<3>(dstIndex, id);
			} else {
				Vec2 arcDir = { 0.0f, 0.0f };
				float cosArcDa = 1.0f, sinArcDa = 0.0f;
				uint32_t numArcPoints = 1;
				if (_LineJoin == LineJoin::Round) {
					const RoundJoinArc& arc = roundJoinArcs[iSegment];
					arcDir = arc.m_ArcDir;
					cosArcDa = arc.m_CosDa;
					sinArcDa = arc.m_SinDa;
					numArcPoints = arc.m_NumArcPoints;
				}

				const uint16_t center = (uint16_t)(firstVertexID + 2);
				*dstPos++ = p1;
				*dstPos++ = vec2Sub(p1, vec2Scale(l01, innerHsw));
				for (uint32_t iArcPoint = 1; iArcPoint < numArcPoints; ++iArcPoint) {
					arcDir = vec2Rotate(arcDir, cosArcDa, sinArcDa);
					*dstPos++ = { p1.x + hsw * arcDir.x, p1.y + hsw * arcDir.y };
				}
				*dstPos++ = vec2Sub(p1, vec2Scale(l12, innerHsw));
				outerFirst = (uint16_t)(firstVertexID + 3);
				outerLast = (uint16_t)(outerFirst + numArcPoints);

				for (uint32_t iArcPoint = 0; iArcPoint < numArcPoints; ++iArcPoint) {
					const uint16_t id[3] = { center, (uint16_t)(outerFirst + iArcPoint), (uint16_t)(outerFirst + iArcPoint + 1) };
					dstIndex = copyIndices<3>(dstIndex, id);
				}

				const uint16_t id[3] = { center, x0, x1 };
				dstIndex = copyIndices<3>(dstIndex, id);
			}

			const uint16_t tL = leftInner ? x0 : outerFirst;
			const uint16_t tR = leftInner ? outerFirst : x0;
			if (prevSegmentLeftID != 0xFFFF) {
				dstIndex = connectJoin2(dstIndex, prevSegmentLeftID, prevSegmentRightID, tL, tR);
			} else {
				firstSegmentLeftID = tL;
				firstSegmentRightID = tR;
			}

			prevSegmentLeftID = leftInner ? x1 : outerLast;
			prevSegmentRightID = leftInner ? outerLast : x1;
			d01 = d12;
			continue;
		}

		if (leftPointProjDist >= 0.0f) {
			// The left point is the inner corner.
			const Vec2 innerCorner = vec2Add(p1, v_hsw);

			if (_LineJoin == LineJoin::Miter || (_LineJoin == LineJoin::Round && roundJoinArcs[iSegment].m_NumArcPoints == 0)) {
				const uint16_t firstVertexID = (uint16_t)(dstPos - posStart);

				Vec2 p[2] = {
					innerCorner,
					vec2Sub(p1, v_hsw)
				};

				dstPos = copyPos<2>(dstPos, &p[0]);

				if (prevSegmentLeftID != 0xFFFF) {
					VG_CHECK(prevSegmentRightID != 0xFFFF, "Invalid previous segment");

					uint16_t id[6] = {
						prevSegmentLeftID, prevSegmentRightID, (uint16_t)(firstVertexID + 1),
						prevSegmentLeftID, (uint16_t)(firstVertexID + 1), firstVertexID
					};

					dstIndex = copyIndices<6>(dstIndex, &id[0]);
				} else {
					firstSegmentLeftID = firstVertexID;
					firstSegmentRightID = firstVertexID + 1;
				}

				prevSegmentLeftID = firstVertexID;
				prevSegmentRightID = firstVertexID + 1;
			} else {
				const Vec2 r01 = vec2PerpCW(d01);
				const Vec2 r12 = vec2PerpCW(d12);

				// Assume _LineJoin == LineJoin::Bevel
				Vec2 arcDir = r01;
				float cosArcDa = 1.0f, sinArcDa = 0.0f;
				uint32_t numArcPoints = 1;
				if (_LineJoin == LineJoin::Round) {
					const RoundJoinArc& arc = roundJoinArcs[iSegment];
					arcDir = arc.m_ArcDir;
					cosArcDa = arc.m_CosDa;
					sinArcDa = arc.m_SinDa;
					numArcPoints = arc.m_NumArcPoints;
				}

				Vec2 p[3] = {
					innerCorner,
					vec2Add(p1, vec2Scale(r01, hsw)),
					vec2Add(p1, vec2Scale(r12, hsw))
				};

				uint16_t firstFanVertexID = (uint16_t)(dstPos - posStart);
				dstPos = copyPos<2>(dstPos, &p[0]);
				for (uint32_t iArcPoint = 1; iArcPoint < numArcPoints; ++iArcPoint) {
					arcDir = vec2Rotate(arcDir, cosArcDa, sinArcDa);

					*dstPos++ = { p1.x + hsw * arcDir.x, p1.y + hsw * arcDir.y };
				}
				*dstPos++ = p[2];

				if (prevSegmentLeftID != 0xFFFF) {
					VG_CHECK(prevSegmentRightID != 0xFFFF, "Invalid previous segment");

					uint16_t id[6] = {
						prevSegmentLeftID, prevSegmentRightID, (uint16_t)(firstFanVertexID + 1),
						prevSegmentLeftID, (uint16_t)(firstFanVertexID + 1), firstFanVertexID
					};

					dstIndex = copyIndices<6>(dstIndex, &id[0]);
				} else {
					firstSegmentLeftID = firstFanVertexID;
					firstSegmentRightID = firstFanVertexID + 1;
				}

				// Generate the triangle fan.
				for (uint32_t iArcPoint = 0; iArcPoint < numArcPoints; ++iArcPoint) {
					const uint16_t idBase = firstFanVertexID + (uint16_t)iArcPoint;
					uint16_t id[3] = {
						firstFanVertexID, (uint16_t)(idBase + 1), (uint16_t)(idBase + 2)
					};
					dstIndex = copyIndices<3>(dstIndex, &id[0]);
				}

				prevSegmentLeftID = firstFanVertexID;
				prevSegmentRightID = firstFanVertexID + (uint16_t)numArcPoints + 1;
			}
		} else {
			// The right point is the inner corner.
			const Vec2 innerCorner = vec2Sub(p1, v_hsw);

			if (_LineJoin == LineJoin::Miter || (_LineJoin == LineJoin::Round && roundJoinArcs[iSegment].m_NumArcPoints == 0)) {
				const uint16_t firstVertexID = (uint16_t)(dstPos - posStart);

				Vec2 p[2] = {
					innerCorner,
					vec2Add(p1, v_hsw),
				};

				dstPos = copyPos<2>(dstPos, &p[0]);

				if (prevSegmentLeftID != 0xFFFF) {
					VG_CHECK(prevSegmentRightID != 0xFFFF, "Invalid previous segment");

					uint16_t id[6] = {
						prevSegmentLeftID, prevSegmentRightID, firstVertexID,
						prevSegmentLeftID, firstVertexID, (uint16_t)(firstVertexID + 1)
					};

					dstIndex = copyIndices<6>(dstIndex, &id[0]);
				} else {
					firstSegmentLeftID = firstVertexID + 1;
					firstSegmentRightID = firstVertexID;
				}

				prevSegmentLeftID = firstVertexID + 1;
				prevSegmentRightID = firstVertexID;
			} else {
				const Vec2 l01 = vec2PerpCCW(d01);
				const Vec2 l12 = vec2PerpCCW(d12);

				// Assume _LineJoin == LineJoin::Bevel
				Vec2 arcDir = l01;
				float cosArcDa = 1.0f, sinArcDa = 0.0f;
				uint32_t numArcPoints = 1;
				if (_LineJoin == LineJoin::Round) {
					const RoundJoinArc& arc = roundJoinArcs[iSegment];
					arcDir = arc.m_ArcDir;
					cosArcDa = arc.m_CosDa;
					sinArcDa = arc.m_SinDa;
					numArcPoints = arc.m_NumArcPoints;
				}

				Vec2 p[3] = {
					innerCorner,
					vec2Add(p1, vec2Scale(l01, hsw)),
					vec2Add(p1, vec2Scale(l12, hsw))
				};

				uint16_t firstFanVertexID = (uint16_t)(dstPos - posStart);
				dstPos = copyPos<2>(dstPos, &p[0]);
				for (uint32_t iArcPoint = 1; iArcPoint < numArcPoints; ++iArcPoint) {
					arcDir = vec2Rotate(arcDir, cosArcDa, sinArcDa);

					*dstPos++ = { p1.x + hsw * arcDir.x, p1.y + hsw * arcDir.y };
				}
				*dstPos++ = p[2];

				if (prevSegmentLeftID != 0xFFFF && prevSegmentRightID != 0xFFFF) {
					uint16_t id[6] = {
						prevSegmentLeftID, prevSegmentRightID, firstFanVertexID,
						prevSegmentLeftID, firstFanVertexID, (uint16_t)(firstFanVertexID + 1)
					};

					dstIndex = copyIndices<6>(dstIndex, &id[0]);
				} else {
					firstSegmentLeftID = firstFanVertexID + 1;
					firstSegmentRightID = firstFanVertexID;
				}

				for (uint32_t iArcPoint = 0; iArcPoint < numArcPoints; ++iArcPoint) {
					const uint16_t idBase = firstFanVertexID + (uint16_t)iArcPoint;
					uint16_t id[3] = {
						firstFanVertexID, (uint16_t)(idBase + 2), (uint16_t)(idBase + 1)
					};
					dstIndex = copyIndices<3>(dstIndex, &id[0]);
				}

				prevSegmentLeftID = firstFanVertexID + (uint16_t)numArcPoints + 1;
				prevSegmentRightID = firstFanVertexID;
			}
		}

		d01 = d12;
	}

	if (!_Closed) {
		// Last segment of an open path
		const Vec2& p1 = vtx[numPathVertices - 1];

		const Vec2 l01 = vec2PerpCCW(d01);

		if (_LineCap == LineCap::Butt || _LineCap == LineCap::Square) {
			const uint16_t curSegmentLeftID = (uint16_t)(dstPos - posStart);
			const Vec2 l01_hsw = vec2Scale(l01, hsw);

			if (_LineCap == LineCap::Butt) {
				dstPos[0] = vec2Add(p1, l01_hsw);
				dstPos[1] = vec2Sub(p1, l01_hsw);
			} else {
				const Vec2 d01_hsw = vec2Scale(d01, hsw);
				dstPos[0] = vec2Add(p1, vec2Add(l01_hsw, d01_hsw));
				dstPos[1] = vec2Sub(p1, vec2Sub(l01_hsw, d01_hsw));
			}
			dstPos += 2;

			dstIndex[0] = prevSegmentLeftID;
			dstIndex[1] = prevSegmentRightID;
			dstIndex[2] = (uint16_t)(curSegmentLeftID + 1);
			dstIndex[3] = prevSegmentLeftID;
			dstIndex[4] = (uint16_t)(curSegmentLeftID + 1);
			dstIndex[5] = curSegmentLeftID;
			dstIndex += 6;
		} else if (_LineCap == LineCap::Round) {
			const uint16_t curSegmentLeftID = (uint16_t)(dstPos - posStart);
			Vec2 capDir = vec2ArcDir(l01);
			for (uint32_t i = 0; i < numPointsHalfCircle; ++i) {
				*dstPos++ = { p1.x + capDir.x * hsw, p1.y + capDir.y * hsw };

				capDir = vec2Rotate(capDir, cosCapDa, -sinCapDa);
			}

			dstIndex[0] = prevSegmentLeftID;
			dstIndex[1] = prevSegmentRightID;
			dstIndex[2] = (uint16_t)(curSegmentLeftID + (numPointsHalfCircle - 1));
			dstIndex[3] = prevSegmentLeftID;
			dstIndex[4] = (uint16_t)(curSegmentLeftID + (numPointsHalfCircle - 1));
			dstIndex[5] = curSegmentLeftID;
			dstIndex += 6;

			for (uint32_t i = 0; i < numPointsHalfCircle - 2; ++i) {
				const uint16_t idBase = curSegmentLeftID + (uint16_t)i;
				dstIndex[0] = curSegmentLeftID;
				dstIndex[1] = (uint16_t)(idBase + 2);
				dstIndex[2] = (uint16_t)(idBase + 1);
				dstIndex += 3;
			}
		}
	} else {
		// Generate the first segment quad.
		dstIndex[0] = prevSegmentLeftID;
		dstIndex[1] = prevSegmentRightID;
		dstIndex[2] = firstSegmentRightID;
		dstIndex[3] = prevSegmentLeftID;
		dstIndex[4] = firstSegmentRightID;
		dstIndex[5] = firstSegmentLeftID;
		dstIndex += 6;
	}

	endGeometry(stroker, &out, dstPos, dstIndex, false, mesh);
}

template<bool _Closed, LineCap::Enum _LineCap, LineJoin::Enum _LineJoin>
void polylineStrokeAA(Stroker* stroker, Mesh* mesh, const Vec2* vtx, uint32_t numPathVertices, float strokeWidth, Color color, const StrokerSink* sink)
{
	const uint32_t numSegments = numPathVertices - (_Closed ? 0 : 1);
	const uint32_t c0 = colorSetAlpha(color, 0);
	const uint32_t c0_c_c_c0[4] = { c0, color, color, c0 };
	const float hsw = (strokeWidth - stroker->m_FringeWidth) * 0.5f;
	const float hsw_aa = hsw + stroker->m_FringeWidth;
	const float da = bx::acos((stroker->m_Scale * hsw) / ((stroker->m_Scale * hsw) + stroker->m_TesselationTolerance)) * 2.0f;
	const uint32_t numPointsHalfCircle = bx::max(2u, (uint32_t)bx::ceil(bx::kPi / da));

	// Round caps are generated by rotating the cap direction by a constant angle.
	float cosCapDa = 1.0f, sinCapDa = 0.0f;
	if (!_Closed && _LineCap == LineCap::Round) {
		const float capDa = bx::kPi / (float)(numPointsHalfCircle - 1);
		cosCapDa = bx::cos(capDa);
		sinCapDa = bx::sin(capDa);
	}

	// Precalculate all segment directions and join extrusion vectors.
	const Vec2* segmentDirs;
	const Vec2* extrusionVecs;
	const uint8_t* innerBevelFlags;
	const uint32_t numInnerBevelJoins = calcSegmentDirsAndExtrusions(stroker, vtx, numPathVertices, _Closed, hsw_aa, &segmentDirs, &extrusionVecs, &innerBevelFlags);

	const uint32_t firstSegmentID = _Closed ? 0 : 1;
	const uint32_t numJoins = numSegments > firstSegmentID ? numSegments - firstSegmentID : 0;

	// Round cap geometry: numPointsHalfCircle * 2 vertices, fan + AA quads indices (+ 18 connecting the last segment).
	const uint32_t numRoundCapIndices = (numPointsHalfCircle - 2) * 3 + (numPointsHalfCircle - 1) * 6;

	// Precalculate the arcs of round joins.
	const RoundJoinArc* roundJoinArcs = nullptr;
	uint32_t totalArcPoints = 0;
	if (_LineJoin == LineJoin::Round) {
		roundJoinArcs = calcRoundJoinArcs(stroker, segmentDirs, extrusionVecs, _Closed ? numPathVertices : numPathVertices - 1, firstSegmentID, numSegments, hsw_aa, da, &totalArcPoints);
	}

	// Calculate the exact amount of geometry and reserve space for it.
	// Joins: miter: 4 vertices + 18 indices, bevel: 6 vertices + 18 + 9 indices, round: numArcPoints * 2 + 4 vertices + 18 + numArcPoints * 9 indices.
	// The 18 indices connect each join to the previous segment. The first join of a closed path is connected by the
	// closing quads instead, so closed paths need no extra indices. Open paths need space for the 2 caps.
	uint32_t numVertices = 0;
	uint32_t numIndices = 0;
	if (_LineJoin == LineJoin::Miter) {
		numVertices = numJoins * 4;
		numIndices = numJoins * 18;
	} else if (_LineJoin == LineJoin::Bevel) {
		numVertices = numJoins * 6;
		numIndices = numJoins * 27;
	} else {
		numVertices = numJoins * 4 + totalArcPoints * 2;
		numIndices = numJoins * 18 + totalArcPoints * 9;
	}

	// Inner bevel joins: miter +2 vertices, bevel/round +3 vertices, +9 indices (see needsInnerBevel()).
	{
		const uint32_t numFanJoins = countInnerBevelFanJoins(innerBevelFlags, numInnerBevelJoins, firstSegmentID, numSegments, _LineJoin, roundJoinArcs);
		numVertices += numInnerBevelJoins * 2 + numFanJoins;
		numIndices += numInnerBevelJoins * 9;
	}

	if (!_Closed) {
		if (_LineCap == LineCap::Round) {
			numVertices += numPointsHalfCircle * 4;
			numIndices += numRoundCapIndices * 2 + 18;
		} else {
			numVertices += 8;
			numIndices += 6 + 24;
		}
	}

	GeometryOutput out;
	beginGeometry(stroker, sink, numVertices, numIndices, true, &out);

	const Vec2* posStart = out.m_Pos;
	Vec2* dstPos = out.m_Pos;
	uint32_t* dstColor = out.m_Color;
	uint16_t* dstIndex = out.m_Index;

	Vec2 d01;
	uint16_t prevSegmentLeftID = 0xFFFF;
	uint16_t prevSegmentLeftAAID = 0xFFFF;
	uint16_t prevSegmentRightID = 0xFFFF;
	uint16_t prevSegmentRightAAID = 0xFFFF;
	uint16_t firstSegmentLeftID = 0xFFFF;
	uint16_t firstSegmentLeftAAID = 0xFFFF;
	uint16_t firstSegmentRightID = 0xFFFF;
	uint16_t firstSegmentRightAAID = 0xFFFF;

	if (!_Closed) {
		// First segment of an open path
		const Vec2& p0 = vtx[0];

		d01 = segmentDirs[0];

		const Vec2 l01 = vec2PerpCCW(d01);

		if (_LineCap == LineCap::Butt || _LineCap == LineCap::Square) {
			const Vec2 l01_hsw = vec2Scale(l01, hsw);
			const Vec2 l01_hsw_aa = vec2Scale(l01, hsw_aa);

			if (_LineCap == LineCap::Butt) {
				const Vec2 d01_aa = vec2Scale(d01, stroker->m_FringeWidth);

				dstPos[0] = vec2Add(p0, vec2Sub(l01_hsw_aa, d01_aa));
				dstPos[1] = vec2Add(p0, l01_hsw);
				dstPos[2] = vec2Sub(p0, l01_hsw);
				dstPos[3] = vec2Sub(p0, vec2Add(l01_hsw_aa, d01_aa));
			} else {
				const Vec2 d01_hsw = vec2Scale(d01, hsw);
				const Vec2 d01_hsw_aa = vec2Scale(d01, hsw_aa);

				dstPos[0] = vec2Add(p0, vec2Sub(l01_hsw_aa, d01_hsw_aa));
				dstPos[1] = vec2Add(p0, vec2Sub(l01_hsw, d01_hsw));
				dstPos[2] = vec2Sub(p0, vec2Add(l01_hsw, d01_hsw));
				dstPos[3] = vec2Sub(p0, vec2Add(l01_hsw_aa, d01_hsw_aa));
			}
			dstPos += 4;
			dstColor = copyColor<4>(dstColor, &c0_c_c_c0[0]);

			dstIndex[0] = 0;
			dstIndex[1] = 2;
			dstIndex[2] = 1;
			dstIndex[3] = 0;
			dstIndex[4] = 3;
			dstIndex[5] = 2;
			dstIndex += 6;

			prevSegmentLeftAAID = 0;
			prevSegmentLeftID = 1;
			prevSegmentRightID = 2;
			prevSegmentRightAAID = 3;
		} else if (_LineCap == LineCap::Round) {
			Vec2 capDir = vec2ArcDir(l01);
			for (uint32_t i = 0; i < numPointsHalfCircle; ++i) {
				dstPos[0] = { p0.x + capDir.x * hsw, p0.y + capDir.y * hsw };
				dstPos[1] = { p0.x + capDir.x * hsw_aa, p0.y + capDir.y * hsw_aa };
				dstPos += 2;
				dstColor = copyColor<2>(dstColor, &c0_c_c_c0[2]);

				capDir = vec2Rotate(capDir, cosCapDa, sinCapDa);
			}

			// Generate indices for the triangle fan
			for (uint32_t i = 0; i < numPointsHalfCircle - 2; ++i) {
				dstIndex[0] = 0;
				dstIndex[1] = (uint16_t)((i << 1) + 2);
				dstIndex[2] = (uint16_t)((i << 1) + 4);
				dstIndex += 3;
			}

			// Generate indices for the AA quads
			for (uint32_t i = 0; i < numPointsHalfCircle - 1; ++i) {
				const uint16_t idBase = (uint16_t)(i << 1);
				dstIndex[0] = idBase;
				dstIndex[1] = (uint16_t)(idBase + 1);
				dstIndex[2] = (uint16_t)(idBase + 3);
				dstIndex[3] = idBase;
				dstIndex[4] = (uint16_t)(idBase + 3);
				dstIndex[5] = (uint16_t)(idBase + 2);
				dstIndex += 6;
			}

			prevSegmentLeftAAID = 1;
			prevSegmentLeftID = 0;
			prevSegmentRightID = (uint16_t)((numPointsHalfCircle - 1) * 2);
			prevSegmentRightAAID = (uint16_t)((numPointsHalfCircle - 1) * 2 + 1);
		} else {
			VG_CHECK(false, "Unknown line cap type");
		}
	} else {
		d01 = segmentDirs[numPathVertices - 1];
	}

	for (uint32_t iSegment = firstSegmentID; iSegment < numSegments; ++iSegment) {
		const Vec2& p1 = vtx[iSegment];
		const Vec2 d12 = segmentDirs[iSegment];
		const Vec2 v = extrusionVecs[iSegment];
		const Vec2 v_hsw_aa = vec2Scale(v, hsw_aa);

		// Check which one of the points is the inner corner.
		float leftPointAAProjDist = d12.x * v_hsw_aa.x + d12.y * v_hsw_aa.y;
		if (innerBevelFlags[iSegment]) {
			// Inner bevel (see polylineStroke()): [X0aa, X0, X1, X1aa], then the outer side: [M, Maa] or
			// [join point, (arc/bevel point, AA point)...]. The inner fringe is a quad over X0-X1.
			const bool leftInner = leftPointAAProjDist >= 0.0f;
			const float innerSign = leftInner ? 1.0f : -1.0f;
			const Vec2 nI01 = vec2Scale(vec2PerpCCW(d01), innerSign);
			const Vec2 nI12 = vec2Scale(vec2PerpCCW(d12), innerSign);
			const uint16_t firstVertexID = (uint16_t)(dstPos - posStart);
			const uint16_t x0aa = firstVertexID;
			const uint16_t x0 = (uint16_t)(firstVertexID + 1);
			const uint16_t x1 = (uint16_t)(firstVertexID + 2);
			const uint16_t x1aa = (uint16_t)(firstVertexID + 3);
			dstPos[0] = vec2Add(p1, vec2Scale(nI01, hsw_aa));
			dstPos[1] = vec2Add(p1, vec2Scale(nI01, hsw));
			dstPos[2] = vec2Add(p1, vec2Scale(nI12, hsw));
			dstPos[3] = vec2Add(p1, vec2Scale(nI12, hsw_aa));
			dstPos += 4;
			dstColor = copyColor<4>(dstColor, &c0_c_c_c0[0]);

			uint16_t outerFirst, outerLast;
			if (_LineJoin == LineJoin::Miter || (_LineJoin == LineJoin::Round && roundJoinArcs[iSegment].m_NumArcPoints == 0)) {
				const Vec2 v_hsw = vec2Scale(v, hsw);
				dstPos[0] = vec2Sub(p1, vec2Scale(v_hsw, innerSign));
				dstPos[1] = vec2Sub(p1, vec2Scale(v_hsw_aa, innerSign));
				dstPos += 2;
				dstColor = copyColor<2>(dstColor, &c0_c_c_c0[2]);
				outerFirst = outerLast = (uint16_t)(firstVertexID + 4);

				const uint16_t id[3] = { x0, outerFirst, x1 };
				dstIndex = copyIndices<3>(dstIndex, id);
			} else {
				Vec2 arcDir = { 0.0f, 0.0f };
				float cosArcDa = 1.0f, sinArcDa = 0.0f;
				uint32_t numArcPoints = 1;
				if (_LineJoin == LineJoin::Round) {
					const RoundJoinArc& arc = roundJoinArcs[iSegment];
					arcDir = arc.m_ArcDir;
					cosArcDa = arc.m_CosDa;
					sinArcDa = arc.m_SinDa;
					numArcPoints = arc.m_NumArcPoints;
				}

				const uint16_t center = (uint16_t)(firstVertexID + 4);
				*dstPos++ = p1;
				*dstColor++ = color;

				const Vec2 nO01 = vec2Scale(nI01, -1.0f);
				const Vec2 nO12 = vec2Scale(nI12, -1.0f);
				const float bevelOffset = _LineJoin == LineJoin::Bevel ? bx::abs(vec2Dot(nO01, nO12)) * stroker->m_FringeWidth : 0.0f;
				dstPos[0] = vec2Sub(vec2Add(p1, vec2Scale(nO01, hsw)), vec2Scale(d01, bevelOffset));
				dstPos[1] = vec2Add(p1, vec2Scale(nO01, hsw_aa));
				dstPos += 2;
				dstColor = copyColor<2>(dstColor, &c0_c_c_c0[2]);
				for (uint32_t iArcPoint = 1; iArcPoint < numArcPoints; ++iArcPoint) {
					arcDir = vec2Rotate(arcDir, cosArcDa, sinArcDa);
					dstPos[0] = vec2Add(p1, vec2Scale(arcDir, hsw));
					dstPos[1] = vec2Add(p1, vec2Scale(arcDir, hsw_aa));
					dstPos += 2;
					dstColor = copyColor<2>(dstColor, &c0_c_c_c0[2]);
				}
				dstPos[0] = vec2Add(vec2Add(p1, vec2Scale(nO12, hsw)), vec2Scale(d12, bevelOffset));
				dstPos[1] = vec2Add(p1, vec2Scale(nO12, hsw_aa));
				dstPos += 2;
				dstColor = copyColor<2>(dstColor, &c0_c_c_c0[2]);
				outerFirst = (uint16_t)(firstVertexID + 5);
				outerLast = (uint16_t)(outerFirst + numArcPoints * 2);

				uint16_t arcID = outerFirst;
				for (uint32_t iArcPoint = 0; iArcPoint < numArcPoints; ++iArcPoint) {
					const uint16_t id[9] = {
						center, arcID, (uint16_t)(arcID + 2),
						arcID, (uint16_t)(arcID + 1), (uint16_t)(arcID + 3),
						arcID, (uint16_t)(arcID + 3), (uint16_t)(arcID + 2)
					};
					dstIndex = copyIndices<9>(dstIndex, id);
					arcID += 2;
				}

				const uint16_t id[3] = { center, x0, x1 };
				dstIndex = copyIndices<3>(dstIndex, id);
			}

			{
				const uint16_t id[6] = { x0aa, x0, x1, x0aa, x1, x1aa };
				dstIndex = copyIndices<6>(dstIndex, id);
			}

			// Outer IDs: solid vertex, AA vertex = solid vertex + 1
			const uint16_t tLA = leftInner ? x0aa : (uint16_t)(outerFirst + 1);
			const uint16_t tL = leftInner ? x0 : outerFirst;
			const uint16_t tR = leftInner ? outerFirst : x0;
			const uint16_t tRA = leftInner ? (uint16_t)(outerFirst + 1) : x0aa;
			if (prevSegmentLeftAAID != 0xFFFF) {
				dstIndex = connectJoin4(dstIndex, prevSegmentLeftAAID, prevSegmentLeftID, prevSegmentRightID, prevSegmentRightAAID, tLA, tL, tR, tRA);
			} else {
				VG_CHECK(_Closed, "Invalid previous segment");
				firstSegmentLeftAAID = tLA;
				firstSegmentLeftID = tL;
				firstSegmentRightID = tR;
				firstSegmentRightAAID = tRA;
			}

			prevSegmentLeftAAID = leftInner ? x1aa : (uint16_t)(outerLast + 1);
			prevSegmentLeftID = leftInner ? x1 : outerLast;
			prevSegmentRightID = leftInner ? outerLast : x1;
			prevSegmentRightAAID = leftInner ? (uint16_t)(outerLast + 1) : x1aa;
			d01 = d12;
			continue;
		}

		if (leftPointAAProjDist >= 0.0f) {
			// The left point is the inner corner.
			const Vec2 v_hsw = vec2Scale(v, hsw);
			const Vec2 innerCornerAA = vec2Add(p1, v_hsw_aa);
			const Vec2 innerCorner = vec2Add(p1, v_hsw);

			if (_LineJoin == LineJoin::Miter || (_LineJoin == LineJoin::Round && roundJoinArcs[iSegment].m_NumArcPoints == 0)) {
				const uint16_t firstVertexID = (uint16_t)(dstPos - posStart);

				Vec2 p[4] = {
					innerCornerAA,
					innerCorner,
					vec2Sub(p1, v_hsw),
					vec2Sub(p1, v_hsw_aa)
				};

				dstPos = copyPos<4>(dstPos, &p[0]);
				dstColor = copyColor<4>(dstColor, &c0_c_c_c0[0]);

				if (prevSegmentLeftAAID != 0xFFFF) {
					VG_CHECK(prevSegmentLeftID != 0xFFFF && prevSegmentRightID != 0xFFFF && prevSegmentRightAAID != 0xFFFF, "Invalid previous segment");

					uint16_t id[18] = {
						prevSegmentLeftAAID, prevSegmentLeftID, (uint16_t)(firstVertexID + 1),
						prevSegmentLeftAAID, (uint16_t)(firstVertexID + 1), firstVertexID,
						prevSegmentLeftID, prevSegmentRightID, (uint16_t)(firstVertexID + 2),
						prevSegmentLeftID, (uint16_t)(firstVertexID + 2), (uint16_t)(firstVertexID + 1),
						prevSegmentRightID, prevSegmentRightAAID, (uint16_t)(firstVertexID + 3),
						prevSegmentRightID, (uint16_t)(firstVertexID + 3), (uint16_t)(firstVertexID + 2)
					};

					dstIndex = copyIndices<18>(dstIndex, &id[0]);
				} else {
					VG_CHECK(_Closed, "Invalid previous segment");
					firstSegmentLeftAAID = firstVertexID; // 0
					firstSegmentLeftID = firstVertexID + 1; // 1
					firstSegmentRightID = firstVertexID + 2; // 2
					firstSegmentRightAAID = firstVertexID + 3; // 3
				}

				prevSegmentLeftAAID = firstVertexID;
				prevSegmentLeftID = firstVertexID + 1;
				prevSegmentRightID = firstVertexID + 2;
				prevSegmentRightAAID = firstVertexID + 3;
			} else {
				const Vec2 r01 = vec2PerpCW(d01);
				const Vec2 r12 = vec2PerpCW(d12);

				// Assume _LineJoin == LineJoin::Bevel
				Vec2 arcDir = r01;
				float cosArcDa = 1.0f, sinArcDa = 0.0f;
				uint32_t numArcPoints = 1;
				if (_LineJoin == LineJoin::Round) {
					const RoundJoinArc& arc = roundJoinArcs[iSegment];
					arcDir = arc.m_ArcDir;
					cosArcDa = arc.m_CosDa;
					sinArcDa = arc.m_SinDa;
					numArcPoints = arc.m_NumArcPoints;
				}

				const uint16_t firstFanVertexID = (uint16_t)(dstPos - posStart);

				Vec2 p[2] = {
					innerCornerAA,
					innerCorner
				};
				dstPos = copyPos<2>(dstPos, &p[0]);
				dstColor = copyColor<2>(dstColor, &c0_c_c_c0[0]);

				// First arc vertex
				{
					Vec2 p[2] = {
						vec2Add(p1, vec2Scale(r01, hsw)),
						vec2Add(p1, vec2Scale(r01, hsw_aa))
					};

					if (_LineJoin == LineJoin::Bevel) {
						const float cosAngle = bx::abs(vec2Dot(r01, r12));
						p[0] = vec2Sub(p[0], vec2Scale(d01, (cosAngle * stroker->m_FringeWidth)));
					}

					dstPos = copyPos<2>(dstPos, &p[0]);
					dstColor = copyColor<2>(dstColor, &c0_c_c_c0[2]);
				}

				// Middle arc vertices
				for (uint32_t iArcPoint = 1; iArcPoint < numArcPoints; ++iArcPoint) {
					arcDir = vec2Rotate(arcDir, cosArcDa, sinArcDa);

					Vec2 p[2] = {
						vec2Add(p1, vec2Scale(arcDir, hsw)),
						vec2Add(p1, vec2Scale(arcDir, hsw_aa))
					};

					dstPos = copyPos<2>(dstPos, &p[0]);
					dstColor = copyColor<2>(dstColor, &c0_c_c_c0[2]);
				}

				// Last arc vertex
				{
					Vec2 p[2] = {
						vec2Add(p1, vec2Scale(r12, hsw)),
						vec2Add(p1, vec2Scale(r12, hsw_aa))
					};

					if (_LineJoin == LineJoin::Bevel) {
						const float cosAngle = bx::abs(vec2Dot(r01, r12));
						p[0] = vec2Add(p[0], vec2Scale(d12, (cosAngle * stroker->m_FringeWidth)));
					}

					dstPos = copyPos<2>(dstPos, &p[0]);
					dstColor = copyColor<2>(dstColor, &c0_c_c_c0[2]);
				}

				if (prevSegmentLeftAAID != 0xFFFF) {
					VG_CHECK(prevSegmentLeftID != 0xFFFF && prevSegmentRightID != 0xFFFF && prevSegmentRightAAID != 0xFFFF, "Invalid previous segment");

					uint16_t id[18] = {
						prevSegmentLeftAAID, prevSegmentLeftID, (uint16_t)(firstFanVertexID + 1),
						prevSegmentLeftAAID, (uint16_t)(firstFanVertexID + 1), firstFanVertexID,
						prevSegmentLeftID, prevSegmentRightID, (uint16_t)(firstFanVertexID + 2),
						prevSegmentLeftID, (uint16_t)(firstFanVertexID + 2), (uint16_t)(firstFanVertexID + 1),
						prevSegmentRightID, prevSegmentRightAAID, (uint16_t)(firstFanVertexID + 3),
						prevSegmentRightID, (uint16_t)(firstFanVertexID + 3), (uint16_t)(firstFanVertexID + 2)
					};

					dstIndex = copyIndices<18>(dstIndex, &id[0]);
				} else {
					VG_CHECK(_Closed, "Invalid previous segment");
					firstSegmentLeftAAID = firstFanVertexID; // 0
					firstSegmentLeftID = firstFanVertexID + 1; // 1
					firstSegmentRightID = firstFanVertexID + 2; // 2
					firstSegmentRightAAID = firstFanVertexID + 3; // 3
				}

				// Generate the slice.
				uint16_t arcID = firstFanVertexID + 2;
				for (uint32_t iArcPoint = 0; iArcPoint < numArcPoints; ++iArcPoint) {
					uint16_t id[9] = {
						(uint16_t)(firstFanVertexID + 1), arcID, (uint16_t)(arcID + 2),
						arcID, (uint16_t)(arcID + 1), (uint16_t)(arcID + 3),
						arcID, (uint16_t)(arcID + 3), (uint16_t)(arcID + 2)
					};
					dstIndex = copyIndices<9>(dstIndex, &id[0]);

					arcID += 2;
				}

				prevSegmentLeftAAID = firstFanVertexID;
				prevSegmentLeftID = firstFanVertexID + 1;
				prevSegmentRightID = arcID;
				prevSegmentRightAAID = arcID + 1;
			}
		} else {
			// The right point is the inner corner.
			const Vec2 v_hsw = vec2Scale(v, hsw);
			const Vec2 innerCornerAA = vec2Sub(p1, v_hsw_aa);
			const Vec2 innerCorner = vec2Sub(p1, v_hsw);

			if (_LineJoin == LineJoin::Miter || (_LineJoin == LineJoin::Round && roundJoinArcs[iSegment].m_NumArcPoints == 0)) {
				const uint16_t firstFanVertexID = (uint16_t)(dstPos - posStart);

				Vec2 p[4] = {
					innerCornerAA,
					innerCorner,
					vec2Add(p1, v_hsw),
					vec2Add(p1, v_hsw_aa)
				};

				dstPos = copyPos<4>(dstPos, &p[0]);
				dstColor = copyColor<4>(dstColor, &c0_c_c_c0[0]);

				if (prevSegmentLeftAAID != 0xFFFF) {
					VG_CHECK(prevSegmentLeftID != 0xFFFF && prevSegmentRightID != 0xFFFF && prevSegmentRightAAID != 0xFFFF, "Invalid previous segment");

					uint16_t id[18] = {
						prevSegmentLeftAAID, prevSegmentLeftID, (uint16_t)(firstFanVertexID + 2),
						prevSegmentLeftAAID, (uint16_t)(firstFanVertexID + 2), (uint16_t)(firstFanVertexID + 3),
						prevSegmentLeftID, prevSegmentRightID, (uint16_t)(firstFanVertexID + 1),
						prevSegmentLeftID, (uint16_t)(firstFanVertexID + 1), (uint16_t)(firstFanVertexID + 2),
						prevSegmentRightID, prevSegmentRightAAID, firstFanVertexID,
						prevSegmentRightID, firstFanVertexID, (uint16_t)(firstFanVertexID + 1)
					};

					dstIndex = copyIndices<18>(dstIndex, &id[0]);
				} else {
					firstSegmentLeftAAID = firstFanVertexID + 3;
					firstSegmentLeftID = firstFanVertexID + 2;
					firstSegmentRightID = firstFanVertexID + 1;
					firstSegmentRightAAID = firstFanVertexID + 0;
				}

				prevSegmentLeftAAID = firstFanVertexID + 3;
				prevSegmentLeftID = firstFanVertexID + 2;
				prevSegmentRightID = firstFanVertexID + 1;
				prevSegmentRightAAID = firstFanVertexID;
			} else {
				const Vec2 l01 = vec2PerpCCW(d01);
				const Vec2 l12 = vec2PerpCCW(d12);

				// Assume _LineJoin == LineJoin::Bevel
				Vec2 arcDir = l01;
				float cosArcDa = 1.0f, sinArcDa = 0.0f;
				uint32_t numArcPoints = 1;
				if (_LineJoin == LineJoin::Round) {
					const RoundJoinArc& arc = roundJoinArcs[iSegment];
					arcDir = arc.m_ArcDir;
					cosArcDa = arc.m_CosDa;
					sinArcDa = arc.m_SinDa;
					numArcPoints = arc.m_NumArcPoints;
				}

				const uint16_t firstFanVertexID = (uint16_t)(dstPos - posStart);

				Vec2 p[2] = {
					innerCornerAA,
					innerCorner
				};
				dstPos = copyPos<2>(dstPos, &p[0]);
				dstColor = copyColor<2>(dstColor, &c0_c_c_c0[0]);

				// First arc vertex
				{
					Vec2 p[2] = {
						vec2Add(p1, vec2Scale(l01, hsw)),
						vec2Add(p1, vec2Scale(l01, hsw_aa))
					};

					if (_LineJoin == LineJoin::Bevel) {
						const float cosAngle = bx::abs(vec2Dot(l01, l12));
						p[0] = vec2Sub(p[0], vec2Scale(d01, (cosAngle * stroker->m_FringeWidth)));
					}

					dstPos = copyPos<2>(dstPos, &p[0]);
					dstColor = copyColor<2>(dstColor, &c0_c_c_c0[2]);
				}

				// Middle arc vertices
				for (uint32_t iArcPoint = 1; iArcPoint < numArcPoints; ++iArcPoint) {
					arcDir = vec2Rotate(arcDir, cosArcDa, sinArcDa);

					Vec2 p[2] = {
						vec2Add(p1, vec2Scale(arcDir, hsw)),
						vec2Add(p1, vec2Scale(arcDir, hsw_aa))
					};

					dstPos = copyPos<2>(dstPos, &p[0]);
					dstColor = copyColor<2>(dstColor, &c0_c_c_c0[2]);
				}

				// Last arc vertex
				{
					Vec2 p[2] = {
						vec2Add(p1, vec2Scale(l12, hsw)),
						vec2Add(p1, vec2Scale(l12, hsw_aa))
					};

					if (_LineJoin == LineJoin::Bevel) {
						const float cosAngle = bx::abs(vec2Dot(l01, l12));
						p[0] = vec2Add(p[0], vec2Scale(d12, (cosAngle * stroker->m_FringeWidth)));
					}

					dstPos = copyPos<2>(dstPos, &p[0]);
					dstColor = copyColor<2>(dstColor, &c0_c_c_c0[2]);
				}

				if (prevSegmentLeftAAID != 0xFFFF) {
					VG_CHECK(prevSegmentLeftID != 0xFFFF && prevSegmentRightID != 0xFFFF && prevSegmentRightAAID != 0xFFFF, "Invalid previous segment");

					uint16_t id[18] = {
						prevSegmentLeftAAID, prevSegmentLeftID, (uint16_t)(firstFanVertexID + 2),
						prevSegmentLeftAAID, (uint16_t)(firstFanVertexID + 2), (uint16_t)(firstFanVertexID + 3),
						prevSegmentLeftID, prevSegmentRightID, (uint16_t)(firstFanVertexID + 1),
						prevSegmentLeftID, (uint16_t)(firstFanVertexID + 1), (uint16_t)(firstFanVertexID + 2),
						prevSegmentRightID, prevSegmentRightAAID, firstFanVertexID,
						prevSegmentRightID, firstFanVertexID, (uint16_t)(firstFanVertexID + 1)
					};

					dstIndex = copyIndices<18>(dstIndex, &id[0]);
				} else {
					firstSegmentLeftAAID = firstFanVertexID + 3;
					firstSegmentLeftID = firstFanVertexID + 2;
					firstSegmentRightID = firstFanVertexID + 1;
					firstSegmentRightAAID = firstFanVertexID + 0;
				}

				// Generate the slice.
				uint16_t arcID = firstFanVertexID + 2;
				for (uint32_t iArcPoint = 0; iArcPoint < numArcPoints; ++iArcPoint) {
					uint16_t id[9] = {
						(uint16_t)(firstFanVertexID + 1), (uint16_t)(arcID + 2), arcID,
						arcID, (uint16_t)(arcID + 3), (uint16_t)(arcID + 1),
						arcID, (uint16_t)(arcID + 2), (uint16_t)(arcID + 3)
					};
					dstIndex = copyIndices<9>(dstIndex, &id[0]);

					arcID += 2;
				}

				prevSegmentLeftAAID = arcID + 1;
				prevSegmentLeftID = arcID;
				prevSegmentRightID = firstFanVertexID + 1;
				prevSegmentRightAAID = firstFanVertexID;
			}
		}

		d01 = d12;
	}

	if (!_Closed) {
		// Last segment of an open path
		const Vec2& p1 = vtx[numPathVertices - 1];

		const Vec2 l01 = vec2PerpCCW(d01);

		if (_LineCap == LineCap::Butt || _LineCap == LineCap::Square) {
			const uint16_t curSegmentLeftAAID = (uint16_t)(dstPos - posStart);
			const Vec2 l01_hsw = vec2Scale(l01, hsw);
			const Vec2 l01_hsw_aa = vec2Scale(l01, hsw_aa);

			if (_LineCap == LineCap::Butt) {
				const Vec2 d01_aa = vec2Scale(d01, stroker->m_FringeWidth);

				dstPos[0] = vec2Add(p1, vec2Add(l01_hsw_aa, d01_aa));
				dstPos[1] = vec2Add(p1, l01_hsw);
				dstPos[2] = vec2Sub(p1, l01_hsw);
				dstPos[3] = vec2Sub(p1, vec2Sub(l01_hsw_aa, d01_aa));
			} else {
				const Vec2 d01_hsw = vec2Scale(d01, hsw);
				const Vec2 d01_hsw_aa = vec2Scale(d01, hsw_aa);

				dstPos[0] = vec2Add(p1, vec2Add(l01_hsw_aa, d01_hsw_aa));
				dstPos[1] = vec2Add(p1, vec2Add(l01_hsw, d01_hsw));
				dstPos[2] = vec2Sub(p1, vec2Sub(l01_hsw, d01_hsw));
				dstPos[3] = vec2Sub(p1, vec2Sub(l01_hsw_aa, d01_hsw_aa));
			}
			dstPos += 4;
			dstColor = copyColor<4>(dstColor, &c0_c_c_c0[0]);

			uint16_t id[24] = {
				prevSegmentLeftAAID, prevSegmentLeftID, (uint16_t)(curSegmentLeftAAID + 1),
				prevSegmentLeftAAID, (uint16_t)(curSegmentLeftAAID + 1), curSegmentLeftAAID,
				prevSegmentLeftID, prevSegmentRightID, (uint16_t)(curSegmentLeftAAID + 2),
				prevSegmentLeftID, (uint16_t)(curSegmentLeftAAID + 2), (uint16_t)(curSegmentLeftAAID + 1),
				prevSegmentRightID, prevSegmentRightAAID, (uint16_t)(curSegmentLeftAAID + 3),
				prevSegmentRightID, (uint16_t)(curSegmentLeftAAID + 3), (uint16_t)(curSegmentLeftAAID + 2),
				curSegmentLeftAAID, (uint16_t)(curSegmentLeftAAID + 1), (uint16_t)(curSegmentLeftAAID + 2),
				curSegmentLeftAAID, (uint16_t)(curSegmentLeftAAID + 2), (uint16_t)(curSegmentLeftAAID + 3)
			};
			dstIndex = copyIndices<24>(dstIndex, &id[0]);
		} else if (_LineCap == LineCap::Round) {
			const uint16_t curSegmentLeftID = (uint16_t)(dstPos - posStart);
			Vec2 capDir = vec2ArcDir(l01);

			for (uint32_t i = 0; i < numPointsHalfCircle; ++i) {
				dstPos[0] = { p1.x + capDir.x * hsw, p1.y + capDir.y * hsw };
				dstPos[1] = { p1.x + capDir.x * hsw_aa, p1.y + capDir.y * hsw_aa };
				dstPos += 2;
				dstColor = copyColor<2>(dstColor, &c0_c_c_c0[2]);

				capDir = vec2Rotate(capDir, cosCapDa, -sinCapDa);
			}

			uint16_t id[18] = {
				prevSegmentLeftAAID, prevSegmentLeftID, curSegmentLeftID,
				prevSegmentLeftAAID, curSegmentLeftID, (uint16_t)(curSegmentLeftID + 1),
				prevSegmentLeftID, prevSegmentRightID, (uint16_t)(curSegmentLeftID + (numPointsHalfCircle - 1) * 2),
				prevSegmentLeftID, (uint16_t)(curSegmentLeftID + (numPointsHalfCircle - 1) * 2), curSegmentLeftID,
				prevSegmentRightID, prevSegmentRightAAID, (uint16_t)(curSegmentLeftID + (numPointsHalfCircle - 1) * 2 + 1),
				prevSegmentRightID, (uint16_t)(curSegmentLeftID + (numPointsHalfCircle - 1) * 2 + 1), (uint16_t)(curSegmentLeftID + (numPointsHalfCircle - 1) * 2)
			};
			dstIndex = copyIndices<18>(dstIndex, &id[0]);

			// Generate indices for the triangle fan
			for (uint32_t i = 0; i < numPointsHalfCircle - 2; ++i) {
				const uint16_t idBase = curSegmentLeftID + (uint16_t)(i << 1);
				dstIndex[0] = curSegmentLeftID;
				dstIndex[1] = (uint16_t)(idBase + 4);
				dstIndex[2] = (uint16_t)(idBase + 2);
				dstIndex += 3;
			}

			// Generate indices for the AA quads
			for (uint32_t i = 0; i < numPointsHalfCircle - 1; ++i) {
				const uint16_t idBase = curSegmentLeftID + (uint16_t)(i << 1);
				dstIndex[0] = idBase;
				dstIndex[1] = (uint16_t)(idBase + 3);
				dstIndex[2] = (uint16_t)(idBase + 1);
				dstIndex[3] = idBase;
				dstIndex[4] = (uint16_t)(idBase + 2);
				dstIndex[5] = (uint16_t)(idBase + 3);
				dstIndex += 6;
			}
		}
	} else {
		VG_CHECK(firstSegmentLeftAAID != 0xFFFF && firstSegmentLeftID != 0xFFFF && firstSegmentRightID != 0xFFFF && firstSegmentRightAAID != 0xFFFF, "Invalid first segment");

		uint16_t id[18] = {
			prevSegmentLeftAAID, prevSegmentLeftID, firstSegmentLeftID,
			prevSegmentLeftAAID, firstSegmentLeftID, firstSegmentLeftAAID,
			prevSegmentLeftID, prevSegmentRightID, firstSegmentRightID,
			prevSegmentLeftID, firstSegmentRightID, firstSegmentLeftID,
			prevSegmentRightID, prevSegmentRightAAID, firstSegmentRightAAID,
			prevSegmentRightID, firstSegmentRightAAID, firstSegmentRightID
		};
		dstIndex = copyIndices<18>(dstIndex, &id[0]);
	}

	endGeometry(stroker, &out, dstPos, dstIndex, true, mesh);
}

template<LineCap::Enum _LineCap, LineJoin::Enum _LineJoin>
void polylineStrokeAAThin(Stroker* stroker, Mesh* mesh, const Vec2* vtx, uint32_t numPathVertices, Color color, bool closed, const StrokerSink* sink)
{
	const uint32_t numSegments = numPathVertices - (closed ? 0 : 1);
	const uint32_t c0 = colorSetAlpha(color, 0);
	const uint32_t c0_c_c0_c0[4] = { c0, color, c0, c0 };
	const float hsw_aa = stroker->m_FringeWidth;

	// Precalculate all segment directions and join extrusion vectors.
	const Vec2* segmentDirs;
	const Vec2* extrusionVecs;
	const uint8_t* innerBevelFlags;
	const uint32_t numInnerBevelJoins = calcSegmentDirsAndExtrusions(stroker, vtx, numPathVertices, closed, hsw_aa, &segmentDirs, &extrusionVecs, &innerBevelFlags);

	const uint32_t firstSegmentID = closed ? 0 : 1;

	// Every join generates a fixed amount of geometry (miter: 3 vertices + 12 indices, bevel: 4 vertices + 12 + 3 indices).
	// The 12 indices connect each join to the previous segment. The first join of a closed path is connected by the
	// closing quads instead, so closed paths need no extra indices. Open paths need space for the 2 caps.
	uint32_t numVertices = 0;
	uint32_t numIndices = 0;
	{
		const uint32_t numJoins = numSegments > firstSegmentID ? numSegments - firstSegmentID : 0;
		numVertices = numJoins * (_LineJoin == LineJoin::Miter ? 3 : 4);
		numIndices = numJoins * (_LineJoin == LineJoin::Miter ? 12 : 15);

		// Inner bevel joins: +1 vertex, +3 indices (see needsInnerBevel()).
		numVertices += numInnerBevelJoins;
		numIndices += numInnerBevelJoins * 3;
		if (!closed) {
			numVertices += 6;
			numIndices += 12;
		}
	}

	GeometryOutput out;
	beginGeometry(stroker, sink, numVertices, numIndices, true, &out);

	const Vec2* posStart = out.m_Pos;
	Vec2* dstPos = out.m_Pos;
	uint32_t* dstColor = out.m_Color;
	uint16_t* dstIndex = out.m_Index;

	Vec2 d01;
	uint16_t prevSegmentLeftAAID = 0xFFFF;
	uint16_t prevSegmentMiddleID = 0xFFFF;
	uint16_t prevSegmentRightAAID = 0xFFFF;

	uint16_t firstSegmentLeftAAID = 0xFFFF;
	uint16_t firstSegmentMiddleID = 0xFFFF;
	uint16_t firstSegmentRightAAID = 0xFFFF;

	if (!closed) {
		// First segment of an open path
		const Vec2& p0 = vtx[0];

		d01 = segmentDirs[0];

		const Vec2 l01 = vec2PerpCCW(d01);

		if (_LineCap == LineCap::Butt || _LineCap == LineCap::Square) {
			const Vec2 l01_hsw_aa = vec2Scale(l01, hsw_aa);

			if (_LineCap == LineCap::Butt) {
				dstPos[0] = vec2Add(p0, l01_hsw_aa);
				dstPos[1] = p0;
				dstPos[2] = vec2Sub(p0, l01_hsw_aa);
			} else {
				const Vec2 d01_hsw_aa = vec2Scale(d01, hsw_aa);

				dstPos[0] = vec2Add(p0, vec2Sub(l01_hsw_aa, d01_hsw_aa));
				dstPos[1] = p0;
				dstPos[2] = vec2Sub(p0, vec2Add(l01_hsw_aa, d01_hsw_aa));
			}
			dstPos += 3;
			dstColor = copyColor<3>(dstColor, &c0_c_c0_c0[0]);

			prevSegmentLeftAAID = 0;
			prevSegmentMiddleID = 1;
			prevSegmentRightAAID = 2;
		} else if (_LineCap == LineCap::Round) {
			VG_CHECK(false, "Round caps not implemented for thin strokes.");
		} else {
			VG_CHECK(false, "Unknown line cap type");
		}
	} else {
		d01 = segmentDirs[numPathVertices - 1];
	}

	for (uint32_t iSegment = firstSegmentID; iSegment < numSegments; ++iSegment) {
		const Vec2& p1 = vtx[iSegment];
		const Vec2 d12 = segmentDirs[iSegment];
		const Vec2 v = extrusionVecs[iSegment];
		const Vec2 v_hsw_aa = vec2Scale(v, hsw_aa);

		// Check which one of the points is the inner corner.
		float leftPointAAProjDist = d12.x * v_hsw_aa.x + d12.y * v_hsw_aa.y;
		if (innerBevelFlags[iSegment]) {
			// Inner bevel (see polylineStroke()): [X0aa, P, X1aa], then the outer side: [Maa] or [O0aa, O1aa].
			const bool leftInner = leftPointAAProjDist >= 0.0f;
			const float innerSign = leftInner ? 1.0f : -1.0f;
			const Vec2 nI01 = vec2Scale(vec2PerpCCW(d01), innerSign);
			const Vec2 nI12 = vec2Scale(vec2PerpCCW(d12), innerSign);
			const uint16_t firstVertexID = (uint16_t)(dstPos - posStart);
			const uint16_t x0aa = firstVertexID;
			const uint16_t mid = (uint16_t)(firstVertexID + 1);
			const uint16_t x1aa = (uint16_t)(firstVertexID + 2);
			dstPos[0] = vec2Add(p1, vec2Scale(nI01, hsw_aa));
			dstPos[1] = p1;
			dstPos[2] = vec2Add(p1, vec2Scale(nI12, hsw_aa));
			dstPos += 3;
			dstColor = copyColor<3>(dstColor, &c0_c_c0_c0[0]);

			uint16_t outerFirst, outerLast;
			if (_LineJoin == LineJoin::Miter) {
				*dstPos++ = vec2Sub(p1, vec2Scale(v_hsw_aa, innerSign));
				*dstColor++ = c0;
				outerFirst = outerLast = (uint16_t)(firstVertexID + 3);
			} else {
				dstPos[0] = vec2Sub(p1, vec2Scale(nI01, hsw_aa));
				dstPos[1] = vec2Sub(p1, vec2Scale(nI12, hsw_aa));
				dstPos += 2;
				dstColor = copyColor<2>(dstColor, &c0_c_c0_c0[2]);
				outerFirst = (uint16_t)(firstVertexID + 3);
				outerLast = (uint16_t)(firstVertexID + 4);

				const uint16_t id[3] = { mid, outerFirst, outerLast };
				dstIndex = copyIndices<3>(dstIndex, id);
			}

			{
				const uint16_t id[3] = { mid, x0aa, x1aa };
				dstIndex = copyIndices<3>(dstIndex, id);
			}

			const uint16_t tLA = leftInner ? x0aa : outerFirst;
			const uint16_t tRA = leftInner ? outerFirst : x0aa;
			if (prevSegmentLeftAAID != 0xFFFF) {
				dstIndex = connectJoin3(dstIndex, prevSegmentLeftAAID, prevSegmentMiddleID, prevSegmentRightAAID, tLA, mid, tRA);
			} else {
				VG_CHECK(closed, "Invalid previous segment");
				firstSegmentLeftAAID = tLA;
				firstSegmentMiddleID = mid;
				firstSegmentRightAAID = tRA;
			}

			prevSegmentLeftAAID = leftInner ? x1aa : outerLast;
			prevSegmentMiddleID = mid;
			prevSegmentRightAAID = leftInner ? outerLast : x1aa;
			d01 = d12;
			continue;
		}

		if (leftPointAAProjDist >= 0.0f) {
			// The left point is the inner corner.
			const Vec2 innerCorner = vec2Add(p1, v_hsw_aa);

			if (_LineJoin == LineJoin::Miter) {
				const uint16_t firstVertexID = (uint16_t)(dstPos - posStart);

				Vec2 p[3] = {
					innerCorner,
					p1,
					vec2Sub(p1, v_hsw_aa)
				};

				dstPos = copyPos<3>(dstPos, &p[0]);
				dstColor = copyColor<3>(dstColor, &c0_c_c0_c0[0]);

				if (prevSegmentLeftAAID != 0xFFFF) {
					VG_CHECK(prevSegmentMiddleID != 0xFFFF && prevSegmentRightAAID != 0xFFFF, "Invalid previous segment");

					uint16_t id[12] = {
						prevSegmentLeftAAID, prevSegmentMiddleID, (uint16_t)(firstVertexID + 1),
						prevSegmentLeftAAID, (uint16_t)(firstVertexID + 1), firstVertexID,
						prevSegmentMiddleID, prevSegmentRightAAID, (uint16_t)(firstVertexID + 2),
						prevSegmentMiddleID, (uint16_t)(firstVertexID + 2), (uint16_t)(firstVertexID + 1)
					};

					dstIndex = copyIndices<12>(dstIndex, id);
				} else {
					VG_CHECK(closed, "Invalid previous segment");
					firstSegmentLeftAAID = firstVertexID;
					firstSegmentMiddleID = firstVertexID + 1;
					firstSegmentRightAAID = firstVertexID + 2;
				}

				prevSegmentLeftAAID = firstVertexID;
				prevSegmentMiddleID = firstVertexID + 1;
				prevSegmentRightAAID = firstVertexID + 2;
			} else {
				VG_CHECK(_LineJoin != LineJoin::Round, "Round joins not implemented for thin strokes.");
				const Vec2 r01 = vec2PerpCW(d01);
				const Vec2 r12 = vec2PerpCW(d12);

				Vec2 p[4] = {
					innerCorner,
					p1,
					vec2Add(p1, vec2Scale(r01, hsw_aa)),
					vec2Add(p1, vec2Scale(r12, hsw_aa))
				};

				const uint16_t firstFanVertexID = (uint16_t)(dstPos - posStart);
				dstPos = copyPos<4>(dstPos, &p[0]);
				dstColor = copyColor<4>(dstColor, &c0_c_c0_c0[0]);

				if (prevSegmentLeftAAID != 0xFFFF) {
					VG_CHECK(prevSegmentMiddleID != 0xFFFF && prevSegmentRightAAID != 0xFFFF, "Invalid previous segment");

					uint16_t id[12] = {
						prevSegmentLeftAAID, prevSegmentMiddleID, (uint16_t)(firstFanVertexID + 1),
						prevSegmentLeftAAID, (uint16_t)(firstFanVertexID + 1), firstFanVertexID,
						prevSegmentMiddleID, prevSegmentRightAAID, (uint16_t)(firstFanVertexID + 2),
						prevSegmentMiddleID, (uint16_t)(firstFanVertexID + 2), (uint16_t)(firstFanVertexID + 1)
					};

					dstIndex = copyIndices<12>(dstIndex, id);
				} else {
					VG_CHECK(closed, "Invalid previous segment");
					firstSegmentLeftAAID = firstFanVertexID;
					firstSegmentMiddleID = firstFanVertexID + 1;
					firstSegmentRightAAID = firstFanVertexID + 2;
				}

				uint16_t id[3] = {
					(uint16_t)(firstFanVertexID + 1), (uint16_t)(firstFanVertexID + 2), (uint16_t)(firstFanVertexID + 3)
				};
				dstIndex = copyIndices<3>(dstIndex, id);

				prevSegmentLeftAAID = firstFanVertexID;
				prevSegmentMiddleID = firstFanVertexID + 1;
				prevSegmentRightAAID = firstFanVertexID + 3;
			}
		} else {
			// The right point is the inner corner.
			const Vec2 innerCorner = vec2Sub(p1, v_hsw_aa);

			if (_LineJoin == LineJoin::Miter) {
				const uint16_t firstFanVertexID = (uint16_t)(dstPos - posStart);

				Vec2 p[3] = {
					innerCorner,
					p1,
					vec2Add(p1, v_hsw_aa)
				};

				dstPos = copyPos<3>(dstPos, &p[0]);
				dstColor = copyColor<3>(dstColor, &c0_c_c0_c0[0]);

				if (prevSegmentLeftAAID != 0xFFFF) {
					VG_CHECK(prevSegmentMiddleID != 0xFFFF && prevSegmentRightAAID != 0xFFFF, "Invalid previous segment");

					uint16_t id[12] = {
						prevSegmentLeftAAID, prevSegmentMiddleID, (uint16_t)(firstFanVertexID + 1),
						prevSegmentLeftAAID, (uint16_t)(firstFanVertexID + 1), (uint16_t)(firstFanVertexID + 2),
						prevSegmentMiddleID, prevSegmentRightAAID, firstFanVertexID,
						prevSegmentMiddleID, firstFanVertexID, (uint16_t)(firstFanVertexID + 1)
					};

					dstIndex = copyIndices<12>(dstIndex, id);
				} else {
					firstSegmentLeftAAID = firstFanVertexID + 2;
					firstSegmentMiddleID = firstFanVertexID + 1;
					firstSegmentRightAAID = firstFanVertexID + 0;
				}

				prevSegmentLeftAAID = firstFanVertexID + 2;
				prevSegmentMiddleID = firstFanVertexID + 1;
				prevSegmentRightAAID = firstFanVertexID;
			} else {
				const Vec2 l01 = vec2PerpCCW(d01);
				const Vec2 l12 = vec2PerpCCW(d12);

				Vec2 p[4] = {
					innerCorner,
					p1,
					vec2Add(p1, vec2Scale(l01, hsw_aa)),
					vec2Add(p1, vec2Scale(l12, hsw_aa))
				};

				const uint16_t firstFanVertexID = (uint16_t)(dstPos - posStart);
				dstPos = copyPos<4>(dstPos, &p[0]);
				dstColor = copyColor<4>(dstColor, &c0_c_c0_c0[0]);

				if (prevSegmentLeftAAID != 0xFFFF) {
					VG_CHECK(prevSegmentMiddleID != 0xFFFF && prevSegmentRightAAID != 0xFFFF, "Invalid previous segment");

					uint16_t id[12] = {
						prevSegmentLeftAAID, prevSegmentMiddleID, (uint16_t)(firstFanVertexID + 1),
						prevSegmentLeftAAID, (uint16_t)(firstFanVertexID + 1), (uint16_t)(firstFanVertexID + 2),
						prevSegmentMiddleID, prevSegmentRightAAID, firstFanVertexID,
						prevSegmentMiddleID, firstFanVertexID, (uint16_t)(firstFanVertexID + 1)
					};

					dstIndex = copyIndices<12>(dstIndex, id);
				} else {
					firstSegmentLeftAAID = firstFanVertexID + 2;
					firstSegmentMiddleID = firstFanVertexID + 1;
					firstSegmentRightAAID = firstFanVertexID + 0;
				}

				uint16_t id[3] = {
					(uint16_t)(firstFanVertexID + 1), (uint16_t)(firstFanVertexID + 3), (uint16_t)(firstFanVertexID + 2)
				};
				dstIndex = copyIndices<3>(dstIndex, id);

				prevSegmentLeftAAID = firstFanVertexID + 3;
				prevSegmentMiddleID = firstFanVertexID + 1;
				prevSegmentRightAAID = firstFanVertexID;
			}
		}

		d01 = d12;
	}

	if (!closed) {
		// Last segment of an open path
		const Vec2& p1 = vtx[numPathVertices - 1];

		const Vec2 l01 = vec2PerpCCW(d01);

		if (_LineCap == LineCap::Butt || _LineCap == LineCap::Square) {
			const uint16_t curSegmentLeftAAID = (uint16_t)(dstPos - posStart);
			const Vec2 l01_hsw_aa = vec2Scale(l01, hsw_aa);

			if (_LineCap == LineCap::Butt) {
				dstPos[0] = vec2Add(p1, l01_hsw_aa);
				dstPos[1] = p1;
				dstPos[2] = vec2Sub(p1, l01_hsw_aa);
			} else {
				const Vec2 d01_hsw = vec2Scale(d01, hsw_aa);

				dstPos[0] = vec2Add(p1, vec2Add(l01_hsw_aa, d01_hsw));
				dstPos[1] = p1;
				dstPos[2] = vec2Sub(p1, vec2Sub(l01_hsw_aa, d01_hsw));
			}
			dstPos += 3;
			dstColor = copyColor<3>(dstColor, &c0_c_c0_c0[0]);

			uint16_t id[12] = {
				prevSegmentLeftAAID, prevSegmentMiddleID, (uint16_t)(curSegmentLeftAAID + 1),
				prevSegmentLeftAAID, (uint16_t)(curSegmentLeftAAID + 1), curSegmentLeftAAID,
				prevSegmentMiddleID, prevSegmentRightAAID, (uint16_t)(curSegmentLeftAAID + 2),
				prevSegmentMiddleID, (uint16_t)(curSegmentLeftAAID + 2), (uint16_t)(curSegmentLeftAAID + 1)
			};
			dstIndex = copyIndices<12>(dstIndex, id);
		} else if (_LineCap == LineCap::Round) {
			VG_CHECK(false, "Round caps not implemented for thin strokes.");
		}
	} else {
		VG_CHECK(firstSegmentLeftAAID != 0xFFFF && firstSegmentMiddleID != 0xFFFF && firstSegmentRightAAID != 0xFFFF, "Invalid first segment");

		uint16_t id[12] = {
			prevSegmentLeftAAID, prevSegmentMiddleID, firstSegmentMiddleID,
			prevSegmentLeftAAID, firstSegmentMiddleID, firstSegmentLeftAAID,
			prevSegmentMiddleID, prevSegmentRightAAID, firstSegmentRightAAID,
			prevSegmentMiddleID, firstSegmentRightAAID, firstSegmentMiddleID
		};
		dstIndex = copyIndices<12>(dstIndex, id);
	}

	endGeometry(stroker, &out, dstPos, dstIndex, true, mesh);
}

inline static void resetGeometry(Stroker* stroker)
{
	stroker->m_NumVertices = 0;
	stroker->m_NumIndices = 0;
}

static void reallocVB(Stroker* stroker, uint32_t n)
{
	// Grow geometrically to avoid O(n^2) reallocations when stroking long paths.
	const uint32_t minCapacity = stroker->m_NumVertices + n;
	stroker->m_VertexCapacity = bx::max<uint32_t>(minCapacity, stroker->m_VertexCapacity + (stroker->m_VertexCapacity >> 1));
	stroker->m_PosBuffer = (Vec2*)bx::alignedRealloc(stroker->m_Allocator, stroker->m_PosBuffer, sizeof(Vec2) * stroker->m_VertexCapacity, 16);
	stroker->m_ColorBuffer = (uint32_t*)bx::alignedRealloc(stroker->m_Allocator, stroker->m_ColorBuffer, sizeof(uint32_t) * stroker->m_VertexCapacity, 16);
}

static BX_FORCE_INLINE void expandVB(Stroker* stroker, uint32_t n)
{
	if (stroker->m_NumVertices + n > stroker->m_VertexCapacity) {
		reallocVB(stroker, n);
	}
}

static void reallocIB(Stroker* stroker, uint32_t n)
{
	const uint32_t minCapacity = stroker->m_NumIndices + n;
	stroker->m_IndexCapacity = bx::max<uint32_t>(minCapacity, stroker->m_IndexCapacity + (stroker->m_IndexCapacity >> 1));
	stroker->m_IndexBuffer = (uint16_t*)bx::alignedRealloc(stroker->m_Allocator, stroker->m_IndexBuffer, sizeof(uint16_t) * stroker->m_IndexCapacity, 16);
}

static BX_FORCE_INLINE void expandIB(Stroker* stroker, uint32_t n)
{
	if (stroker->m_NumIndices + n > stroker->m_IndexCapacity) {
		reallocIB(stroker, n);
	}
}
}
