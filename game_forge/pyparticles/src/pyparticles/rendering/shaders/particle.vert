#version 330 core

// Instanced Attributes
layout (location = 0) in vec2 in_pos;
layout (location = 1) in vec3 in_color;
layout (location = 2) in float in_radius;
layout (location = 3) in float in_freq;
layout (location = 4) in float in_amp;
layout (location = 5) in float in_angle; // Rotation

// Uniforms
uniform vec2 window_size;
uniform float scale;
uniform vec2 offset;

// Output to Geometry Shader
out VS_OUT {
    vec3 color;
    float radius;
    float freq;
    float amp;
    float angle;
} vs_out;

void main() {
    // Transform World -> Screen Pixels -> NDC
    // pos is [-1, 1] in physics world.
    // Screen: x_px = pos.x * scale + offset.x
    // NDC: (x_px / w) * 2 - 1
    
    vec2 screen_pos = in_pos * scale + offset;
    
    // Y-flip for screen coords (Physics +Y is Up? No, canvas logic: sy = h - (y*s + off))
    // Standard GL: +Y is Up.
    // Pygame: +Y is Down.
    // Let's stick to GL coords for simplicity? 
    // Or match Canvas logic exactly to overlay UI.
    // Canvas: sy = H - (phys.y * scale + off.y)
    
    float sy = window_size.y - (in_pos.y * scale + offset.y);
    screen_pos.y = sy;
    
    // Normalize to [-1, 1]
    vec2 ndc_pos = (screen_pos / window_size) * 2.0 - 1.0;
    ndc_pos.y = -ndc_pos.y; // Flip back to GL
    
    gl_Position = vec4(ndc_pos, 0.0, 1.0);
    
    vs_out.color = in_color;
    vs_out.radius = in_radius; // Physical radius
    vs_out.freq = in_freq;
    vs_out.amp = in_amp;
    vs_out.angle = in_angle;
}
