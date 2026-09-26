#include <vg/path.h>
#include <bx/allocator.h>

namespace vg
{
// Caches the last (radius, scale, tolerance) -> (numPoints, cos(dtheta), sin(dtheta)) computation
// so that consecutive shapes with the same radius don't recompute acos/cos/sin. The key is compared
// bitwise and includes every input of the computation, so the result is bit-identical to computing
// it from scratch (no invalidation needed on pathReset()). m_NumPoints == 0 means empty (the computed
// value is always >= 2).
struct ArcStepCache
{
	uint32_t m_KeyRadius;
	uint32_t m_KeyScale;
	uint32_t m_KeyTolerance;
	uint32_t m_NumPoints;
	float m_CosDTheta;
	float m_SinDTheta;
};

struct Path
{
	bx::AllocatorI* m_Allocator;
	float* m_Vertices;
	SubPath* m_SubPaths;
	SubPath* m_CurSubPath;
	uint32_t m_NumVertices;
	uint32_t m_VertexCapacity;
	uint32_t m_NumSubPaths;
	uint32_t m_SubPathCapacity;
	float m_Scale;
	float m_TesselationTolerance;
	ArcStepCache m_RoundedRectCache;        // pathRoundedRect()
	ArcStepCache m_RoundedRectVaryingCache; // pathRoundedRectVarying() (per corner)
	ArcStepCache m_EllipseCache;            // pathEllipse()/pathCircle()
};

static float* pathAllocVertices(Path* path, uint32_t n);
static void pathAddVertex(Path* path, float x, float y);

static inline bool arcStepCacheLookup(const ArcStepCache* cache, uint32_t kr, uint32_t ks, uint32_t kt)
{
	return cache->m_NumPoints != 0
		&& cache->m_KeyRadius == kr
		&& cache->m_KeyScale == ks
		&& cache->m_KeyTolerance == kt
		;
}

static inline void arcStepCacheStore(ArcStepCache* cache, uint32_t kr, uint32_t ks, uint32_t kt, uint32_t numPoints, float cos_dtheta, float sin_dtheta)
{
	cache->m_KeyRadius = kr;
	cache->m_KeyScale = ks;
	cache->m_KeyTolerance = kt;
	cache->m_NumPoints = numPoints;
	cache->m_CosDTheta = cos_dtheta;
	cache->m_SinDTheta = sin_dtheta;
}

// pathRoundedRect(): number of points of a quarter circle (incl. both end points) and the per-step rotation.
static const ArcStepCache* pathGetRoundedRectStep(Path* path, float r)
{
	ArcStepCache* cache = &path->m_RoundedRectCache;
	const uint32_t kr = bx::floatToBits(r);
	const uint32_t ks = bx::floatToBits(path->m_Scale);
	const uint32_t kt = bx::floatToBits(path->m_TesselationTolerance);
	if (!arcStepCacheLookup(cache, kr, ks, kt)) {
		const float da = bx::acos((path->m_Scale * r) / ((path->m_Scale * r) + path->m_TesselationTolerance)) * 2.0f;
		const uint32_t numPointsHalfCircle = bx::max(2, (uint32_t)bx::ceil(bx::kPi / da));
		const uint32_t numPointsQuarterCircle = (numPointsHalfCircle >> 1) + 1;

		const float dtheta = -bx::kPiHalf / (float)(numPointsQuarterCircle - 1);
		arcStepCacheStore(cache, kr, ks, kt, numPointsQuarterCircle, bx::cos(dtheta), bx::sin(dtheta));
	}

	return cache;
}

// pathRoundedRectVarying(): same as above for a single corner (note: slightly different expression, kept as is).
static const ArcStepCache* pathGetRoundedRectVaryingStep(Path* path, float r)
{
	ArcStepCache* cache = &path->m_RoundedRectVaryingCache;
	const uint32_t kr = bx::floatToBits(r);
	const uint32_t ks = bx::floatToBits(path->m_Scale);
	const uint32_t kt = bx::floatToBits(path->m_TesselationTolerance);
	if (!arcStepCacheLookup(cache, kr, ks, kt)) {
		const float halfDa = bx::acos((path->m_Scale * r) / ((path->m_Scale * r) + path->m_TesselationTolerance));
		const uint32_t numPointsHalfCircle = bx::max(2, (uint32_t)bx::ceil(bx::kPiHalf / halfDa));
		const uint32_t numPointsQuarterCircle = (numPointsHalfCircle >> 1) + 1;

		const float dtheta = -bx::kPiHalf / (float)(numPointsQuarterCircle - 1);
		arcStepCacheStore(cache, kr, ks, kt, numPointsQuarterCircle, bx::cos(dtheta), bx::sin(dtheta));
	}

	return cache;
}

// pathEllipse(): number of points of the full circle and the per-step rotation.
static const ArcStepCache* pathGetEllipseStep(Path* path, float avgR)
{
	ArcStepCache* cache = &path->m_EllipseCache;
	const uint32_t kr = bx::floatToBits(avgR);
	const uint32_t ks = bx::floatToBits(path->m_Scale);
	const uint32_t kt = bx::floatToBits(path->m_TesselationTolerance);
	if (!arcStepCacheLookup(cache, kr, ks, kt)) {
		const float da = bx::acos((path->m_Scale * avgR) / ((path->m_Scale * avgR) + path->m_TesselationTolerance)) * 2.0f;

		const uint32_t numPointsHalfCircle = bx::max(2, (uint32_t)bx::ceil(bx::kPi / da));
		const uint32_t numPoints = (numPointsHalfCircle * 2);

		const float dtheta = -bx::kPi2 / (float)numPoints;
		arcStepCacheStore(cache, kr, ks, kt, numPoints, bx::cos(dtheta), bx::sin(dtheta));
	}

	return cache;
}

Path* createPath(bx::AllocatorI* allocator)
{
	Path* path = (Path*)bx::alloc(allocator, sizeof(Path));
	bx::memSet(path, 0, sizeof(Path));
	path->m_Allocator = allocator;
	path->m_Scale = 1.0f;
	path->m_TesselationTolerance = 0.25f;
	return path;
}

void destroyPath(Path* path)
{
	bx::AllocatorI* allocator = path->m_Allocator;

    if (path->m_Vertices) {
        bx::alignedFree(allocator, path->m_Vertices, 16);
    }
	bx::free(allocator, path->m_SubPaths);
	bx::free(allocator, path);
}

void pathReset(Path* path, float scale, float tesselationTolerance)
{
	path->m_Scale = scale;
	path->m_TesselationTolerance = tesselationTolerance;

	if (path->m_SubPathCapacity == 0) {
		path->m_SubPathCapacity = 16;
		path->m_SubPaths = (SubPath*)bx::alloc(path->m_Allocator, sizeof(SubPath) * path->m_SubPathCapacity);
	}

	path->m_NumSubPaths = 0;
	path->m_NumVertices = 0;
	path->m_SubPaths[0].m_IsClosed = false;
	path->m_SubPaths[0].m_NumVertices = 0;
	path->m_SubPaths[0].m_FirstVertexID = 0;
	path->m_CurSubPath = nullptr;
}

void pathMoveTo(Path* path, float x, float y)
{
	if (!path->m_CurSubPath || path->m_CurSubPath->m_NumVertices != 0) {
		// Move on to the next sub path.
		if (path->m_NumSubPaths + 1 > path->m_SubPathCapacity) {
			path->m_SubPathCapacity = bx::max<uint32_t>(path->m_SubPathCapacity + 16, path->m_SubPathCapacity + (path->m_SubPathCapacity >> 1));
			path->m_SubPaths = (SubPath*)bx::realloc(path->m_Allocator, path->m_SubPaths, sizeof(SubPath) * path->m_SubPathCapacity);
		}

		path->m_CurSubPath = &path->m_SubPaths[path->m_NumSubPaths++];
		path->m_CurSubPath->m_IsClosed = false;
		path->m_CurSubPath->m_NumVertices = 0;
		path->m_CurSubPath->m_FirstVertexID = path->m_NumVertices;
	}

	pathAddVertex(path, x, y);
}

// If there's no current sub-path (e.g. a lineTo() right after beginPath()), start a new one at (x, y).
// Same as the HTML canvas "ensure there is a subpath" step; previously this dereferenced a null sub-path.
static inline void pathEnsureSubPath(Path* path, float x, float y)
{
	if (!path->m_CurSubPath || path->m_CurSubPath->m_NumVertices == 0) {
		pathMoveTo(path, x, y);
	}
}

void pathLineTo(Path* path, float x, float y)
{
	pathEnsureSubPath(path, x, y);
	pathAddVertex(path, x, y);
}

void pathCubicTo(Path* path, float c1x, float c1y, float c2x, float c2y, float x, float y)
{
	pathEnsureSubPath(path, c1x, c1y);

	const int MAX_LEVELS = 10;
	float stack[MAX_LEVELS * 8];
	int stackLevel[MAX_LEVELS]; // Subdivision level of each pushed sibling

	VG_CHECK(!path->m_CurSubPath->m_IsClosed, "Cannot add new vertices to a closed path");

	const uint32_t lastVertexID = path->m_CurSubPath->m_FirstVertexID + (path->m_CurSubPath->m_NumVertices - 1);
	const float* lastVertex = &path->m_Vertices[lastVertexID << 1];

	float x1 = lastVertex[0];
	float y1 = lastVertex[1];
	float x2 = c1x;
	float y2 = c1y;
	float x3 = c2x;
	float y3 = c2y;
	float x4 = x;
	float y4 = y;

	const float tessTol = path->m_TesselationTolerance / (path->m_Scale * path->m_Scale);

	// Emitted points are written directly into the vertex buffer (equivalent to pathAddVertex(), incl.
	// its dedup check against the previous vertex of the sub-path and its capacity growth policy),
	// keeping the buffer pointer, vertex count, capacity and last vertex in locals. The counts are
	// committed once at the end.
	float* vertices = path->m_Vertices;
	const uint32_t firstNewVertex = path->m_NumVertices;
	uint32_t numVertices = firstNewVertex;
	uint32_t vertexCapacity = path->m_VertexCapacity;
	float lastX = x1;
	float lastY = y1;

	// NOTE: The subdivision level is tracked explicitly (like NanoVG's recursive version). The number of
	// pushed siblings is <= the current level, so the stack can never overflow.
	int level = 0;
	float* stackPtr = stack;
	for (;;) {
		const float dx = x4 - x1;
		const float dy = y4 - y1;
		const float d2 = bx::abs((x2 - x4) * dy - (y2 - y4) * dx);
		const float d3 = bx::abs((x3 - x4) * dy - (y3 - y4) * dx);
		const float d23 = d2 + d3;

		// Emit the end point when the segment is flat enough or when the max subdivision level
		// has been reached (in the latter case the end point is still emitted, so the vertex isn't lost).
		const bool isFlat = d23 * d23 <= tessTol * (dx * dx + dy * dy);
		const bool isMaxLevel = level >= MAX_LEVELS;
		if (isFlat | isMaxLevel) {
			// Same as pathAddVertex(path, x4, y4)
			const float ddx = lastX - x4;
			const float ddy = lastY - y4;
			const float distSqr = ddx * ddx + ddy * ddy;
			if (!(distSqr < VG_EPSILON)) {
				float* v;
				if (numVertices == vertexCapacity) {
					// Buffer full: grow it exactly like pathAddVertex() would.
					path->m_NumVertices = numVertices;
					v = pathAllocVertices(path, 1);
					vertices = path->m_Vertices;
					vertexCapacity = path->m_VertexCapacity;
				} else {
					v = &vertices[numVertices << 1];
				}

				v[0] = x4;
				v[1] = y4;
				++numVertices;
				lastX = x4;
				lastY = y4;
			}

			// Pop sibling off the stack and decrease level...
			if (stackPtr == stack) {
				break;
			}

			stackPtr -= 8;
			level = stackLevel[(stackPtr - stack) / 8];
			y4 = stackPtr[0];
			x4 = stackPtr[1];
			y3 = stackPtr[2];
			x3 = stackPtr[3];
			y2 = stackPtr[4];
			x2 = stackPtr[5];
			y1 = stackPtr[6];
			x1 = stackPtr[7];
		} else {
			const float x12 = (x1 + x2) * 0.5f;
			const float y12 = (y1 + y2) * 0.5f;
			const float x23 = (x2 + x3) * 0.5f;
			const float y23 = (y2 + y3) * 0.5f;
			const float x34 = (x3 + x4) * 0.5f;
			const float y34 = (y3 + y4) * 0.5f;
			const float x123 = (x12 + x23) * 0.5f;
			const float y123 = (y12 + y23) * 0.5f;
			const float x234 = (x23 + x34) * 0.5f;
			const float y234 = (y23 + y34) * 0.5f;
			const float x1234 = (x123 + x234) * 0.5f;
			const float y1234 = (y123 + y234) * 0.5f;

			// Push sibling on the stack...
			stackPtr[0] = y4;
			stackPtr[1] = x4;
			stackPtr[2] = y34;
			stackPtr[3] = x34;
			stackPtr[4] = y234;
			stackPtr[5] = x234;
			stackPtr[6] = y1234;
			stackPtr[7] = x1234;
			++level;
			stackLevel[(stackPtr - stack) / 8] = level;
			stackPtr += 8;

//			x1 = x1; // NOP
//			y1 = y1; // NOP
			x2 = x12;
			y2 = y12;
			x3 = x123;
			y3 = y123;
			x4 = x1234;
			y4 = y1234;
		}
	}

	path->m_NumVertices = numVertices;
	path->m_CurSubPath->m_NumVertices += numVertices - firstNewVertex;
}

void pathQuadraticTo(Path* path, float cx, float cy, float x, float y)
{
	// Convert quadratic bezier to cubic bezier (http://fontforge.github.io/bezier.html)
	pathEnsureSubPath(path, cx, cy);

	const uint32_t lastVertexID = path->m_CurSubPath->m_FirstVertexID + (path->m_CurSubPath->m_NumVertices - 1);
	const float* lastVertex = &path->m_Vertices[lastVertexID << 1];

	const float x0 = lastVertex[0];
	const float y0 = lastVertex[1];

	const float c1x = x0 + (2.0f / 3.0f) * (cx - x0);
	const float c1y = y0 + (2.0f / 3.0f) * (cy - y0);
	const float c2x = x + (2.0f / 3.0f) * (cx - x);
	const float c2y = y + (2.0f / 3.0f) * (cy - y);

	pathCubicTo(path, c1x, c1y, c2x, c2y, x, y);
}

void pathArcTo(Path* path, float x1, float y1, float x2, float y2, float r)
{
	pathEnsureSubPath(path, x1, y1);

	const uint32_t lastVertexID = path->m_CurSubPath->m_FirstVertexID + (path->m_CurSubPath->m_NumVertices - 1);
	const float* lastVertex = &path->m_Vertices[lastVertexID << 1];

	// nvgArcTo()
	const float x0 = lastVertex[0];
	const float y0 = lastVertex[1];

	// Handle degenerate cases (same as nvgArcTo(); NanoVG's distTol is 0.01 pixels).
	{
		const float distTol = 0.01f / path->m_Scale;
		const float distTolSqr = distTol * distTol;

		const float d01x = x1 - x0, d01y = y1 - y0;
		const float d12x = x2 - x1, d12y = y2 - y1;

		// Squared distance of (x1, y1) from the segment (x0, y0)-(x2, y2) (nvg__distPtSeg())
		const float pqx = x2 - x0, pqy = y2 - y0;
		const float d = pqx * pqx + pqy * pqy;
		float t = pqx * (x1 - x0) + pqy * (y1 - y0);
		if (d > 0.0f) {
			t /= d;
		}
		t = bx::clamp<float>(t, 0.0f, 1.0f);
		const float sx = x0 + t * pqx - x1;
		const float sy = y0 + t * pqy - y1;

		if (d01x * d01x + d01y * d01y < distTolSqr
			|| d12x * d12x + d12y * d12y < distTolSqr
			|| sx * sx + sy * sy < distTolSqr
			|| r < distTol) {
			pathLineTo(path, x1, y1);
			return;
		}
	}

	// Calculate tangential circle to lines (x0,y0)-(x1,y1) and (x1,y1)-(x2,y2).
	float dx0 = x0 - x1;
	float dy0 = y0 - y1;
	float dx1 = x2 - x1;
	float dy1 = y2 - y1;
//	nvg__normalize(&dx0, &dy0);
	{
		const float lenSqr = dx0 * dx0 + dy0 * dy0;
		const float invLen = lenSqr < VG_EPSILON ? 0.0f : bx::rsqrt(lenSqr);
		dx0 *= invLen;
		dy0 *= invLen;
	}

//	nvg__normalize(&dx1, &dy1);
	{
		const float lenSqr = dx1 * dx1 + dy1 * dy1;
		const float invLen = lenSqr < VG_EPSILON ? 0.0f : bx::rsqrt(lenSqr);
		dx1 *= invLen;
		dy1 *= invLen;
	}

	// NOTE: The dot product is clamped because rounding can push it slightly outside [-1, 1] for (anti)parallel
	// directions, which made acos() return NaN (and the whole arc NaN).
	const float a = bx::acos(bx::clamp<float>(dx0 * dx1 + dy0 * dy1, -1.0f, 1.0f));
	const float d = r / bx::tan(a / 2.0f);

	if (d > 10000.0f) {
		pathLineTo(path, x1, y1);
		return;
	}

	float cx, cy, a0, a1;
	Winding::Enum dir;
	const float cross = dx1 * dy0 - dx0 * dy1;
	if (cross > 0.0f) {
		cx = x1 + dx0 * d + dy0 * r;
		cy = y1 + dy0 * d - dx0 * r;

		a0 = bx::atan2(dx0, -dy0);
		a1 = bx::atan2(-dx1, dy1);
		dir = Winding::CW;
	} else {
		cx = x1 + dx0 * d - dy0 * r;
		cy = y1 + dy0 * d + dx0 * r;

		a0 = bx::atan2(-dx0, dy0);
		a1 = bx::atan2(dx1, -dy1);
		dir = Winding::CCW;
	}

	pathArc(path, cx, cy, r, a0, a1, dir);
}

void pathRect(Path* path, float x, float y, float w, float h)
{
	if (bx::abs(w) < VG_EPSILON || bx::abs(h) < VG_EPSILON) {
		return;
	}

	pathMoveTo(path, x, y);
	pathLineTo(path, x, y + h);
	pathLineTo(path, x + w, y + h);
	pathLineTo(path, x + w, y);
	pathClose(path);
}

void pathRoundedRect(Path* path, float x, float y, float w, float h, float r)
{
	if (r < 0.1f) {
		pathRect(path, x, y, w, h);
		return;
	}

	const float rx = bx::min<float>(r, bx::abs(w) * 0.5f) * bx::sign(w);
	const float ry = bx::min<float>(r, bx::abs(h) * 0.5f) * bx::sign(h);

	r = bx::min<float>(rx, ry);

	const ArcStepCache* step = pathGetRoundedRectStep(path, r);
	const uint32_t numPointsQuarterCircle = step->m_NumPoints;
	const float cos_dtheta = step->m_CosDTheta;
	const float sin_dtheta = step->m_SinDTheta;

	pathMoveTo(path, x, y + r);
	pathLineTo(path, x, y + h - r);

	// Bottom left quarter circle
	{
		const float cx = x + r;
		const float cy = y + h - r;
		float* circleVertices = pathAllocVertices(path, numPointsQuarterCircle - 1);

		float ca = -1.0f; // cosf(-PI);
		float sa = 0.0f;  // sinf(-PI);
		for (uint32_t i = 1; i < numPointsQuarterCircle; ++i) {
			const float ns = sin_dtheta * ca + cos_dtheta * sa;
			const float nc = cos_dtheta * ca - sin_dtheta * sa;
			ca = nc;
			sa = ns;

			circleVertices[0] = cx + r * ca;
			circleVertices[1] = cy + r * sa;
			circleVertices += 2;
		}
		path->m_CurSubPath->m_NumVertices += (numPointsQuarterCircle - 1);
	}

	pathLineTo(path, x + w - r, y + h);

	// Bottom right quarter circle
	{
		const float cx = x + w - r;
		const float cy = y + h - r;
		float* circleVertices = pathAllocVertices(path, numPointsQuarterCircle - 1);

		float ca = 0.0f; // cosf(-1.5f * PI);
		float sa = 1.0f; // sinf(-1.5f * PI);
		for (uint32_t i = 1; i < numPointsQuarterCircle; ++i) {
			const float ns = sin_dtheta * ca + cos_dtheta * sa;
			const float nc = cos_dtheta * ca - sin_dtheta * sa;
			ca = nc;
			sa = ns;

			circleVertices[0] = cx + r * ca;
			circleVertices[1] = cy + r * sa;
			circleVertices += 2;
		}
		path->m_CurSubPath->m_NumVertices += (numPointsQuarterCircle - 1);
	}

	pathLineTo(path, x + w, y + r);

	// Top right quarter circle
	{
		const float cx = x + w - r;
		const float cy = y + r;
		float* circleVertices = pathAllocVertices(path, numPointsQuarterCircle - 1);

		float ca = 1.0f; // cosf(0.0f);
		float sa = 0.0f; // sinf(0.0f);
		for (uint32_t i = 1; i < numPointsQuarterCircle; ++i) {
			const float ns = sin_dtheta * ca + cos_dtheta * sa;
			const float nc = cos_dtheta * ca - sin_dtheta * sa;
			ca = nc;
			sa = ns;

			circleVertices[0] = cx + r * ca;
			circleVertices[1] = cy + r * sa;
			circleVertices += 2;
		}
		path->m_CurSubPath->m_NumVertices += (numPointsQuarterCircle - 1);
	}

	pathLineTo(path, x + r, y);

	// Top left quarter circle
	{
		const float cx = x + r;
		const float cy = y + r;
		float* circleVertices = pathAllocVertices(path, numPointsQuarterCircle - 1);

		float ca = 0.0f; // cosf(-0.5f * PI);
		float sa = -1.0f; // sinf(-0.5f * PI);
		for (uint32_t i = 1; i < numPointsQuarterCircle; ++i) {
			const float ns = sin_dtheta * ca + cos_dtheta * sa;
			const float nc = cos_dtheta * ca - sin_dtheta * sa;
			ca = nc;
			sa = ns;

			circleVertices[0] = cx + r * ca;
			circleVertices[1] = cy + r * sa;
			circleVertices += 2;
		}
		path->m_CurSubPath->m_NumVertices += (numPointsQuarterCircle - 1);
	}

	pathClose(path);
}

void pathRoundedRectVarying(Path* path, float x, float y, float w, float h, float rTopLeft, float rTopRight, float rBottomRight, float rBottomLeft)
{
	if (rTopLeft < 0.1f && rBottomLeft < 0.1f && rBottomRight < 0.1f && rTopRight < 0.1f) {
		pathRect(path, x, y, w, h);
		return;
	}

	const float halfw = w * 0.5f;
	const float halfh = h * 0.5f;

	const float rtl = bx::min<float>(rTopLeft, halfw, halfh);
	const float rtr = bx::min<float>(rTopRight, halfw, halfh);
	const float rbl = bx::min<float>(rBottomLeft, halfw, halfh);
	const float rbr = bx::min<float>(rBottomRight, halfw, halfh);

	// Top left corner
	if (rtl < 0.1f) {
		pathMoveTo(path, x, y);
	} else {
		pathMoveTo(path, x + rtl, y);

		const ArcStepCache* step = pathGetRoundedRectVaryingStep(path, rtl);
		const uint32_t numPointsQuarterCircle = step->m_NumPoints;
		const float cos_dtheta = step->m_CosDTheta;
		const float sin_dtheta = step->m_SinDTheta;

		const float cx = x + rtl;
		const float cy = y + rtl;
		float* circleVertices = pathAllocVertices(path, numPointsQuarterCircle - 1);

		float ca = 0.0f;
		float sa = -1.0f;
		for (uint32_t i = 1; i < numPointsQuarterCircle; ++i) {
			const float ns = sin_dtheta * ca + cos_dtheta * sa;
			const float nc = cos_dtheta * ca - sin_dtheta * sa;
			ca = nc;
			sa = ns;

			circleVertices[0] = cx + rtl * ca;
			circleVertices[1] = cy + rtl * sa;
			circleVertices += 2;
		}
		path->m_CurSubPath->m_NumVertices += (numPointsQuarterCircle - 1);
	}

	// Bottom left corner
	if (rbl < 0.1f) {
		pathLineTo(path, x, y + h);
	} else {
		pathLineTo(path, x, y + h - rbl);

		const ArcStepCache* step = pathGetRoundedRectVaryingStep(path, rbl);
		const uint32_t numPointsQuarterCircle = step->m_NumPoints;
		const float cos_dtheta = step->m_CosDTheta;
		const float sin_dtheta = step->m_SinDTheta;

		const float cx = x + rbl;
		const float cy = y + h - rbl;
		float* circleVertices = pathAllocVertices(path, numPointsQuarterCircle - 1);

		float ca = -1.0f;
		float sa = 0.0f;
		for (uint32_t i = 1; i < numPointsQuarterCircle; ++i) {
			const float ns = sin_dtheta * ca + cos_dtheta * sa;
			const float nc = cos_dtheta * ca - sin_dtheta * sa;
			ca = nc;
			sa = ns;

			circleVertices[0] = cx + rbl * ca;
			circleVertices[1] = cy + rbl * sa;
			circleVertices += 2;
		}
		path->m_CurSubPath->m_NumVertices += (numPointsQuarterCircle - 1);
	}

	// Bottom right corner
	if (rbr < 0.1f) {
		pathLineTo(path, x + w, y + h);
	} else {
		pathLineTo(path, x + w - rbr, y + h);

		const ArcStepCache* step = pathGetRoundedRectVaryingStep(path, rbr);
		const uint32_t numPointsQuarterCircle = step->m_NumPoints;
		const float cos_dtheta = step->m_CosDTheta;
		const float sin_dtheta = step->m_SinDTheta;

		const float cx = x + w - rbr;
		const float cy = y + h - rbr;
		float* circleVertices = pathAllocVertices(path, numPointsQuarterCircle - 1);

		float ca = 0.0f; // cosf(-1.5f * PI);
		float sa = 1.0f; // sinf(-1.5f * PI);
		for (uint32_t i = 1; i < numPointsQuarterCircle; ++i) {
			const float ns = sin_dtheta * ca + cos_dtheta * sa;
			const float nc = cos_dtheta * ca - sin_dtheta * sa;
			ca = nc;
			sa = ns;

			circleVertices[0] = cx + rbr * ca;
			circleVertices[1] = cy + rbr * sa;
			circleVertices += 2;
		}
		path->m_CurSubPath->m_NumVertices += (numPointsQuarterCircle - 1);
	}

	// Top right corner
	if (rtr < 0.1f) {
		pathLineTo(path, x + w, y);
	} else {
		pathLineTo(path, x + w, y + rtr);

		const ArcStepCache* step = pathGetRoundedRectVaryingStep(path, rtr);
		const uint32_t numPointsQuarterCircle = step->m_NumPoints;
		const float cos_dtheta = step->m_CosDTheta;
		const float sin_dtheta = step->m_SinDTheta;

		const float cx = x + w - rtr;
		const float cy = y + rtr;
		float* circleVertices = pathAllocVertices(path, numPointsQuarterCircle - 1);

		float ca = 1.0f;
		float sa = 0.0f;
		for (uint32_t i = 1; i < numPointsQuarterCircle; ++i) {
			const float ns = sin_dtheta * ca + cos_dtheta * sa;
			const float nc = cos_dtheta * ca - sin_dtheta * sa;
			ca = nc;
			sa = ns;

			circleVertices[0] = cx + rtr * ca;
			circleVertices[1] = cy + rtr * sa;
			circleVertices += 2;
		}
		path->m_CurSubPath->m_NumVertices += (numPointsQuarterCircle - 1);
	}

	pathClose(path);
}

void pathCircle(Path* path, float cx, float cy, float r)
{
#if 1
	pathEllipse(path, cx, cy, r, r);
#else
	const float da = bx::acos((path->m_Scale * r) / ((path->m_Scale * r) + path->m_TesselationTolerance)) * 2.0f;

	const uint32_t numPointsHalfCircle = bx::max(2, (uint32_t)bx::ceil(bx::kPi / da));
	const uint32_t numPoints = (numPointsHalfCircle * 2);

	pathMoveTo(path, cx + r, cy);

	float* circleVertices = pathAllocVertices(path, numPoints - 1);

	// http://www.iquilezles.org/www/articles/sincos/sincos.htm
	const float dtheta = -bx::kPi2 / (float)numPoints;
	const float cos_dtheta = bx::cos(dtheta);
	const float sin_dtheta = bx::sin(dtheta);

	float ca = 1.0f;
	float sa = 0.0f;
	for (uint32_t i = 1; i < numPoints; ++i) {
		const float nextSin = sin_dtheta * ca + cos_dtheta * sa;
		const float nextCos = cos_dtheta * ca - sin_dtheta * sa;
		ca = nextCos;
		sa = nextSin;

		circleVertices[0] = cx + r * ca;
		circleVertices[1] = cy + r * sa;
		circleVertices += 2;
	}

	path->m_CurSubPath->m_NumVertices += (numPoints - 1);

	pathClose(path);
#endif
}

void pathEllipse(Path* path, float cx, float cy, float rx, float ry)
{
	const float avgR = (rx + ry) * 0.5f;
	const ArcStepCache* step = pathGetEllipseStep(path, avgR);
	const uint32_t numPoints = step->m_NumPoints;
	const float cos_dtheta = step->m_CosDTheta;
	const float sin_dtheta = step->m_SinDTheta;

	pathMoveTo(path, cx + rx, cy);

	float* circleVertices = pathAllocVertices(path, numPoints - 1);

	float ca = 1.0f;
	float sa = 0.0f;
	for (uint32_t i = 1; i < numPoints; ++i) {
		const float nextSin = sin_dtheta * ca + cos_dtheta * sa;
		const float nextCos = cos_dtheta * ca - sin_dtheta * sa;
		ca = nextCos;
		sa = nextSin;

		circleVertices[0] = cx + rx * ca;
		circleVertices[1] = cy + ry * sa;
		circleVertices += 2;
	}

	path->m_CurSubPath->m_NumVertices += (numPoints - 1);

	pathClose(path);
}

void pathArc(Path* path, float cx, float cy, float r, float a0, float a1, Winding::Enum dir)
{
	// a0 and a1 are CW angles from the x axis independent of the selected direction of the arc.
	// Make sure a0 is always less than a1 and they are both inside the [0, 2*Pi] circle.
	while (a0 > bx::kPi2) {
		a0 -= bx::kPi2;
	}
	while (a1 > bx::kPi2) {
		a1 -= bx::kPi2;
	}

	if (dir == Winding::CCW) {
		while (a0 < a1) {
			a0 += bx::kPi2;
		}
	} else {
		while (a1 < a0) {
			a1 += bx::kPi2;
		}
	}

	const float da = bx::acos((path->m_Scale * r) / ((path->m_Scale * r) + path->m_TesselationTolerance)) * 2.0f;
	const uint32_t numPoints = bx::max(2, (uint32_t)bx::ceil(bx::abs(a1 - a0) / da));

	const float dtheta = (a1 - a0) / (float)numPoints;
	const float cos_dtheta = bx::cos(dtheta);
	const float sin_dtheta = bx::sin(dtheta);
	float ca = bx::cos(a0);
	float sa = bx::sin(a0);

	if (path->m_CurSubPath && path->m_CurSubPath->m_NumVertices != 0) {
		pathLineTo(path, cx + r * ca, cy + r * sa);
	} else {
		pathMoveTo(path, cx + r * ca, cy + r * sa);
	}

	float* circleVertices = pathAllocVertices(path, numPoints);
	for (uint32_t i = 0; i < numPoints; ++i) {
		const float nextSin = sin_dtheta * ca + cos_dtheta * sa;
		const float nextCos = cos_dtheta * ca - sin_dtheta * sa;
		ca = nextCos;
		sa = nextSin;

		circleVertices[0] = cx + r * ca;
		circleVertices[1] = cy + r * sa;
		circleVertices += 2;
	}

	path->m_CurSubPath->m_NumVertices += numPoints;
}

void pathPolyline(Path* path, const float* coords, uint32_t numPoints)
{
	if (numPoints == 0) {
		return;
	}

	// If there's no current sub-path (e.g. polyline() right after beginPath()), the first point
	// starts a new one (same as moveTo()).
	if (!path->m_CurSubPath || path->m_CurSubPath->m_NumVertices == 0) {
		pathMoveTo(path, coords[0], coords[1]);
		coords += 2;
		numPoints--;
		if (numPoints == 0) {
			return;
		}
	}

	VG_CHECK(!path->m_CurSubPath->m_IsClosed, "Cannot add new vertices to a closed path");

	if (path->m_CurSubPath->m_NumVertices > 0) {
		const uint32_t lastVertexID = path->m_CurSubPath->m_FirstVertexID + (path->m_CurSubPath->m_NumVertices - 1);
		const float* lastVertex = &path->m_Vertices[lastVertexID << 1];

		const float dx = lastVertex[0] - coords[0];
		const float dy = lastVertex[1] - coords[1];
		const float distSqr = dx * dx + dy * dy;
		if (distSqr < VG_EPSILON) {
			coords += 2;
			numPoints--;
		}
	}

	float* vertices = pathAllocVertices(path, numPoints);
	bx::memCopy(vertices, coords, sizeof(float) * 2 * numPoints);
	path->m_CurSubPath->m_NumVertices += numPoints;
}

void pathClose(Path* path)
{
	if (!path->m_CurSubPath || path->m_CurSubPath->m_IsClosed || path->m_CurSubPath->m_NumVertices <= 2) {
		return;
	}

	path->m_CurSubPath->m_IsClosed = true;

	const float* firstVertex = &path->m_Vertices[path->m_CurSubPath->m_FirstVertexID << 1];
	const float* lastVertex = &path->m_Vertices[(path->m_CurSubPath->m_FirstVertexID + (path->m_CurSubPath->m_NumVertices - 1)) << 1];

	const float dx = lastVertex[0] - firstVertex[0];
	const float dy = lastVertex[1] - firstVertex[1];
	const float distSqr = dx * dx + dy * dy;
	if (distSqr < VG_EPSILON) {
		--path->m_CurSubPath->m_NumVertices;
		--path->m_NumVertices;
	}
}

const float* pathGetVertices(const Path* path)
{
	return path->m_Vertices;
}

uint32_t pathGetNumVertices(const Path* path)
{
	return path->m_NumVertices;
}

const SubPath* pathGetSubPaths(const Path* path)
{
	return path->m_SubPaths;
}

uint32_t pathGetNumSubPaths(const Path* path)
{
	return path->m_NumSubPaths;
}

static float* pathAllocVertices(Path* path, uint32_t n)
{
	if (path->m_NumVertices + n > path->m_VertexCapacity) {
		path->m_VertexCapacity = bx::max(path->m_VertexCapacity + n, path->m_VertexCapacity != 0 ? (path->m_VertexCapacity * 3) >> 1 : 16);
		path->m_Vertices = (float*)bx::alignedRealloc(path->m_Allocator, path->m_Vertices, sizeof(float) * 2 * path->m_VertexCapacity, 16);
	}

	float* p = &path->m_Vertices[path->m_NumVertices << 1];
	path->m_NumVertices += n;

	return p;
}

static void pathAddVertex(Path* path, float x, float y)
{
	// Don't allow adding new vertices to a closed sub-path.
	VG_CHECK(path->m_CurSubPath, "No path");
	VG_CHECK(!path->m_CurSubPath->m_IsClosed, "Cannot add new vertices to a closed path");

	if (path->m_CurSubPath->m_NumVertices != 0) {
		const uint32_t lastVertexID = path->m_CurSubPath->m_FirstVertexID + (path->m_CurSubPath->m_NumVertices - 1);
		const float* lastVertex = &path->m_Vertices[lastVertexID << 1];

		const float dx = lastVertex[0] - x;
		const float dy = lastVertex[1] - y;
		const float distSqr = dx * dx + dy * dy;
		if (distSqr < VG_EPSILON) {
			return;
		}
	}

	float* v = pathAllocVertices(path, 1);
	v[0] = x;
	v[1] = y;

	path->m_CurSubPath->m_NumVertices++;
}
}
