#version 330 core

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

in VS_OUT {
    vec3 color;
    float radius;
    float freq;
    float amp;
    float angle;
    float energy;
} gs_in[];

out vec2 uv;
out vec3 color;
out float radius;
out float freq;
out float amp;
out float angle;
out float energy;

uniform vec2 window_size;
uniform float scale;

void main() {
    float phys_r = gs_in[0].radius + abs(gs_in[0].amp);
    float size_px = phys_r * scale * 2.5; 
    
    vec2 size_ndc = (vec2(size_px) / window_size) * 2.0;
    
    vec4 center = gl_in[0].gl_Position;
    
    color = gs_in[0].color;
    radius = gs_in[0].radius * scale;
    freq = gs_in[0].freq;
    amp = gs_in[0].amp * scale;
    angle = gs_in[0].angle;
    energy = gs_in[0].energy;
    
    gl_Position = center + vec4(-size_ndc.x, -size_ndc.y, 0.0, 0.0);
    uv = vec2(-1.0, -1.0);
    EmitVertex();
    
    gl_Position = center + vec4(size_ndc.x, -size_ndc.y, 0.0, 0.0);
    uv = vec2(1.0, -1.0);
    EmitVertex();
    
    gl_Position = center + vec4(-size_ndc.x, size_ndc.y, 0.0, 0.0);
    uv = vec2(-1.0, 1.0);
    EmitVertex();
    
    gl_Position = center + vec4(size_ndc.x, size_ndc.y, 0.0, 0.0);
    uv = vec2(1.0, 1.0);
    EmitVertex();
    
    EndPrimitive();
}