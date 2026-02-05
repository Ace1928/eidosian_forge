#version 330 core

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in VS_OUT {
    vec3 color;
    float radius;
    float freq;
    float amp;
    float angle;
} gs_in[];

out vec2 uv;
out vec3 color;
out float radius;
out float freq;
out float amp;
out float angle;

uniform vec2 window_size;
uniform float scale;

void main() {
    // We need to emit a quad large enough to cover the wave (radius + amp)
    // Scale radius to NDC
    
    float phys_r = gs_in[0].radius + abs(gs_in[0].amp);
    float size_px = phys_r * scale * 2.5; // Extra margin
    
    vec2 size_ndc = size_px / window_size; // Size in NDC (0-1 range actually?)
    // NDC is -1 to 1. width=2.
    // So size_ndc = (size_px / w) * 2.
    size_ndc = size_ndc * 2.0;
    
    vec4 center = gl_in[0].gl_Position;
    
    // Passthrough
    color = gs_in[0].color;
    radius = gs_in[0].radius * scale; // Pixel radius
    freq = gs_in[0].freq;
    amp = gs_in[0].amp * scale; // Pixel amp
    angle = gs_in[0].angle;
    
    // Quad
    // BL
    gl_Position = center + vec4(-size_ndc.x, -size_ndc.y, 0.0, 0.0);
    uv = vec2(-1.0, -1.0);
    EmitVertex();
    
    // BR
    gl_Position = center + vec4(size_ndc.x, -size_ndc.y, 0.0, 0.0);
    uv = vec2(1.0, -1.0);
    EmitVertex();
    
    // TL
    gl_Position = center + vec4(-size_ndc.x, size_ndc.y, 0.0, 0.0);
    uv = vec2(-1.0, 1.0);
    EmitVertex();
    
    // TR
    gl_Position = center + vec4(size_ndc.x, size_ndc.y, 0.0, 0.0);
    uv = vec2(1.0, 1.0);
    EmitVertex();
    
    EndPrimitive();
}
