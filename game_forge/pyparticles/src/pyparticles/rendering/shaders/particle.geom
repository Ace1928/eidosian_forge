#version 330 core

// ============================================================================
// EIDOSIAN PYPARTICLES V6 - GEOMETRY SHADER
// High-performance billboard expansion with LOD and culling
// ============================================================================

layout (points) in;
layout (triangle_strip, max_vertices = 4) out;

// Input from vertex shader
in VS_OUT {
    vec3 color;
    float radius;
    float freq;
    float amp;
    float angle;
    float energy;
    float depth;
} gs_in[];

// Output to fragment shader
out vec2 uv;
out vec3 frag_color;
out float frag_radius;
out float frag_freq;
out float frag_amp;
out float frag_angle;
out float frag_energy;
out float frag_depth;
out vec2 frag_center;    // Center in NDC for distance calculations

// Uniforms
uniform vec2 window_size;
uniform float view_scale;
uniform float particle_scale;    // Global particle size multiplier
uniform int render_mode;         // 0=normal, 1=wave_only, 2=energy_only
uniform float time;

// Constants
const float BASE_SIZE_MULT = 3.0;    // Base size multiplier
const float MIN_PIXEL_SIZE = 2.0;    // Minimum size in pixels
const float MAX_PIXEL_SIZE = 200.0;  // Maximum size in pixels

void main() {
    vec4 center = gl_in[0].gl_Position;
    
    // Early culling: skip particles far outside view
    if (abs(center.x) > 1.5 || abs(center.y) > 1.5) {
        return;
    }
    
    // Calculate physical radius including wave amplitude
    float base_r = gs_in[0].radius;
    float wave_amp = abs(gs_in[0].amp);
    float phys_r = base_r + wave_amp;
    
    // Apply global particle scale
    phys_r *= particle_scale;
    
    // Convert to NDC size
    float aspect = window_size.x / window_size.y;
    float size_ndc = phys_r * BASE_SIZE_MULT;
    
    // LOD: reduce detail for small particles
    float pixel_size = size_ndc * window_size.y * 0.5;
    
    // Clamp to min/max sizes
    if (pixel_size < MIN_PIXEL_SIZE) {
        size_ndc = MIN_PIXEL_SIZE / (window_size.y * 0.5);
    } else if (pixel_size > MAX_PIXEL_SIZE) {
        size_ndc = MAX_PIXEL_SIZE / (window_size.y * 0.5);
    }
    
    // Aspect-corrected quad size
    vec2 size = vec2(size_ndc / aspect, size_ndc);
    
    // Pass through attributes (flat interpolation would be faster but need smooth for some)
    frag_color = gs_in[0].color;
    frag_radius = base_r;
    frag_freq = gs_in[0].freq;
    frag_amp = gs_in[0].amp;
    frag_angle = gs_in[0].angle;
    frag_energy = gs_in[0].energy;
    frag_depth = gs_in[0].depth;
    frag_center = center.xy;
    
    // Emit quad vertices (triangle strip: BL, BR, TL, TR)
    // Bottom-left
    gl_Position = center + vec4(-size.x, -size.y, 0.0, 0.0);
    uv = vec2(-1.0, -1.0);
    EmitVertex();
    
    // Bottom-right
    gl_Position = center + vec4(size.x, -size.y, 0.0, 0.0);
    uv = vec2(1.0, -1.0);
    EmitVertex();
    
    // Top-left
    gl_Position = center + vec4(-size.x, size.y, 0.0, 0.0);
    uv = vec2(-1.0, 1.0);
    EmitVertex();
    
    // Top-right
    gl_Position = center + vec4(size.x, size.y, 0.0, 0.0);
    uv = vec2(1.0, 1.0);
    EmitVertex();
    
    EndPrimitive();
}