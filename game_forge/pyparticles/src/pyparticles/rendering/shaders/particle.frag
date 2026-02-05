#version 330 core

in vec2 uv;
in vec3 color;
in float radius;
in float freq;
in float amp;
in float angle;

out vec4 fragColor;

void main() {
    // Polar Coords
    float dist = length(uv); // 0 to ~1 (quad edge)
    // Map uv to physical distance?
    // Quad was size_px/2.
    // Let's assume uv length 1 is the boundary of the quad.
    // The quad size was (radius + amp) * 2.5
    // So uv=1 means physical distance = (radius+amp)*2.5.
    
    float scale_factor = 2.5;
    float r_norm = dist * scale_factor * (radius + amp);
    
    float theta = atan(uv.y, uv.x); // -pi to pi
    
    // Rotate by particle angle
    float local_theta = theta - angle;
    
    // Wave Function
    float shape_r = radius + amp * cos(freq * local_theta);
    
    // SDF (Approximate)
    // distance from shape boundary
    // d < 0 inside
    float d = r_norm - shape_r;
    
    // Soft Edge (Anti-aliasing)
    // d is in pixels? No, r_norm is in pixels.
    // Change per pixel is 1.0.
    float alpha = 1.0 - smoothstep(-1.0, 1.0, d);
    
    // Core Glow
    float glow = exp(-d * 0.1);
    
    if (alpha <= 0.0) discard;
    
    // Visuals
    vec3 col = color;
    
    // Highlight perimeter
    float perimeter = 1.0 - smoothstep(0.0, 2.0, abs(d));
    col += vec3(1.0) * perimeter * 0.8;
    
    // Inner gradient
    col *= 0.5 + 0.5 * (1.0 - dist);
    
    fragColor = vec4(col, alpha);
}
