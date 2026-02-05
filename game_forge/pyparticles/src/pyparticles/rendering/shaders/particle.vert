#version 330 core

layout (location = 0) in vec2 in_pos;
layout (location = 1) in vec3 in_color;
layout (location = 2) in float in_radius;
layout (location = 3) in float in_freq;
layout (location = 4) in float in_amp;
layout (location = 5) in float in_angle;
layout (location = 6) in float in_energy;

uniform vec2 window_size;
uniform float scale;
uniform vec2 offset;

out VS_OUT {
    vec3 color;
    float radius;
    float freq;
    float amp;
    float angle;
    float energy;
} vs_out;

void main() {
    vec2 screen_pos = in_pos * scale + offset;
    float sy = window_size.y - (in_pos.y * scale + offset.y);
    screen_pos.y = sy;
    
    vec2 ndc_pos = (screen_pos / window_size) * 2.0 - 1.0;
    ndc_pos.y = -ndc_pos.y;
    
    gl_Position = vec4(ndc_pos, 0.0, 1.0);
    
    vs_out.color = in_color;
    vs_out.radius = in_radius;
    vs_out.freq = in_freq;
    vs_out.amp = in_amp;
    vs_out.angle = in_angle;
    vs_out.energy = in_energy;
}