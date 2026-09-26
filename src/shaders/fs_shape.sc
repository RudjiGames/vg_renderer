$input v_color0, v_shape0, v_shape1, v_shape2

// Analytic (rounded) rectangles and circles, filled or stroked (see createDrawCommand_Shape() in vg.cpp).
// v_shape0.xy: position relative to the center of the shape (in pixels, along the axes of the shape)
// v_shape0.zw: half size
// v_shape1: corner radii { (+x, +y), (+x, -y), (-x, +y), (-x, -y) }
// v_shape2: { half stroke width, AA fringe width, mode (0: fill, 1: stroke, 2: thin stroke), sharp corners }
// The coverage matches the AA fringes of the generated geometry: a linear ramp of the fringe width centered on
// the edge, and a triangular profile of the fringe width for thin strokes.

#include <bgfx_shader.sh>

void main()
{
	vec2 p = v_shape0.xy;
	vec2 radii = p.x > 0.0 ? v_shape1.xy : v_shape1.zw;
	float r = p.y > 0.0 ? radii.x : radii.y;

	vec2 q = abs(p) - v_shape0.zw + r;
	float outside = (r == 0.0 && v_shape2.w > 0.5) ? max(max(q.x, q.y), 0.0) : length(max(q, 0.0) );
	float d = min(max(q.x, q.y), 0.0) + outside - r;

	float aa = v_shape2.y;
	float coverage;
	if (v_shape2.z < 0.5) {
		coverage = clamp(0.5 - d / aa, 0.0, 1.0);
	} else if (v_shape2.z < 1.5) {
		coverage = clamp( (v_shape2.x + 0.5 * aa - abs(d) ) / aa, 0.0, 1.0);
	} else {
		coverage = clamp(1.0 - abs(d) / aa, 0.0, 1.0);
	}

	gl_FragColor = vec4(v_color0.xyz, v_color0.w * coverage);
}
