#version 330 core

// ============================================================================
// EIDOSIAN PYPARTICLES V6 - VERTEX SHADER
// Advanced particle rendering with configurable transforms
// ============================================================================

// Vertex attributes (per-particle instance data)
layout (location = 0) in vec2 in_pos;       // World position [-1, 1]
layout (location = 1) in vec3 in_color;     // RGB color [0, 1]
layout (location = 2) in float in_radius;   // Base radius in world units
layout (location = 3) in float in_freq;     // Wave frequency
layout (location = 4) in float in_amp;      // Wave amplitude
layout (location = 5) in float in_angle;    // Rotation angle (radians)
layout (location = 6) in float in_energy;   // Energy for glow (speed * scale)

// Uniforms for view transforms
uniform vec2 window_size;      // Window dimensions in pixels
uniform vec2 view_center;      // View center in world coords (for panning)
uniform float view_scale;      // Zoom level (1.0 = default, 2.0 = 2x zoom)
uniform float time;            // Animation time for effects

// Output to geometry shader
out VS_OUT {
    vec3 color;
    float radius;
    float freq;
    float amp;
    float angle;
    float energy;
    float depth;        // For future depth sorting
} vs_out;

void main() {
    // Apply view transform: pan and zoom
    vec2 world_pos = in_pos - view_center;
    world_pos *= view_scale;
    
    // Aspect ratio correction
    float aspect = window_size.x / window_size.y;
    vec2 ndc_pos = world_pos;
    ndc_pos.x /= aspect;
    
    // Clamp to prevent particles from going too far off-screen
    // (optimization: geometry shader can cull these)
    
    gl_Position = vec4(ndc_pos, 0.0, 1.0);
    
    // Pass attributes to geometry shader
    vs_out.color = in_color;
    vs_out.radius = in_radius * view_scale;  // Scale radius with zoom
    vs_out.freq = in_freq;
    vs_out.amp = in_amp * view_scale;        // Scale amplitude with zoom
    vs_out.angle = in_angle;
    vs_out.energy = in_energy;
    vs_out.depth = 0.5 + in_pos.y * 0.5;     // Simple depth from Y position
}