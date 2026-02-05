#version 330 core

// ============================================================================
// EIDOSIAN PYPARTICLES V6 - FRAGMENT SHADER
// Advanced particle rendering with wave mechanics, energy visualization,
// and multiple render modes
// ============================================================================

// Inputs from geometry shader
in vec2 uv;
in vec3 frag_color;
in float frag_radius;
in float frag_freq;
in float frag_amp;
in float frag_angle;
in float frag_energy;
in float frag_depth;
in vec2 frag_center;

// Output
out vec4 out_color;

// Uniforms
uniform vec2 window_size;
uniform float time;
uniform int render_mode;           // 0=full, 1=wave, 2=energy, 3=minimal
uniform float glow_intensity;      // Glow strength multiplier
uniform float edge_sharpness;      // Edge definition (higher = sharper)
uniform vec3 ambient_color;        // Background ambient for glow blending
uniform int wave_viz_mode;         // 0=shape, 1=interference, 2=phase

// Constants
const float PI = 3.14159265359;
const float TAU = 6.28318530718;
const float EDGE_AA_WIDTH = 0.08;  // Anti-aliasing edge width
const float GLOW_FALLOFF = 3.0;    // Glow decay rate

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Fast approximate atan2
float fast_atan2(float y, float x) {
    float ax = abs(x);
    float ay = abs(y);
    float a = min(ax, ay) / (max(ax, ay) + 1e-10);
    float s = a * a;
    float r = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a;
    if (ay > ax) r = 1.57079637 - r;
    if (x < 0.0) r = 3.14159265 - r;
    if (y < 0.0) r = -r;
    return r;
}

// Smooth minimum for blending
float smin(float a, float b, float k) {
    float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

// Hash function for noise
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// ============================================================================
// WAVE SHAPE FUNCTIONS
// ============================================================================

// Calculate wave-deformed radius at given angle
float wave_radius(float theta, float base_r, float freq, float amp, float phase) {
    // Primary wave
    float wave = amp * cos(freq * theta + phase);
    
    // Optional: Add harmonics for more complex shapes
    // float harmonic = amp * 0.3 * cos(freq * 2.0 * theta + phase * 1.5);
    
    return base_r + wave;
}

// Calculate wave slope (derivative) for lighting
float wave_slope(float theta, float freq, float amp, float phase) {
    return -amp * freq * sin(freq * theta + phase);
}

// ============================================================================
// RENDER MODES
// ============================================================================

// Standard particle with wave deformation - CALMER colors
vec4 render_standard(float dist, float theta, float local_theta) {
    // Calculate wave-deformed shape
    float shape_r = wave_radius(local_theta, frag_radius, frag_freq, frag_amp, 0.0);
    
    // Normalize for UV space
    float total_r = frag_radius + abs(frag_amp);
    if (total_r < 0.001) total_r = 0.001;
    
    float norm_dist = dist * total_r;
    float norm_shape = shape_r;
    
    // Signed distance from edge
    float d = norm_dist - norm_shape;
    
    // Anti-aliased edge
    float edge_width = EDGE_AA_WIDTH * total_r;
    float alpha = 1.0 - smoothstep(-edge_width, edge_width, d);
    
    if (alpha < 0.01) discard;
    
    // Energy factor - CLAMPED for stability
    float e = clamp(frag_energy * 0.3 + 0.7, 0.5, 1.5);
    
    // Base color - desaturated for calmer look
    vec3 col = frag_color * 0.8;
    
    // Subtle core brightness (not overwhelming glow)
    float core_dist = dist / (total_r + 0.01);
    float core_brightness = (1.0 - core_dist * 0.5) * e * 0.3;
    col += vec3(core_brightness);
    
    // Subtle edge highlight only
    float edge_highlight = exp(-abs(d) * 10.0 / total_r) * 0.15;
    col += vec3(1.0) * edge_highlight;
    
    // Minimal wave crest visualization
    if (abs(frag_amp) > 0.001) {
        float wave_phase = cos(frag_freq * local_theta);
        float crest_intensity = max(0.0, wave_phase) * 0.15 * e;
        col += vec3(0.1, 0.2, 0.3) * crest_intensity * (1.0 - core_dist);
    }
    
    // REMOVED: outer glow, HDR tonemapping, heat shift
    // Keep colors clean and predictable
    
    return vec4(col, alpha * 0.95);
}

// Wave-only visualization (debug/analysis mode)
vec4 render_wave_only(float dist, float theta, float local_theta) {
    float shape_r = wave_radius(local_theta, frag_radius, frag_freq, frag_amp, 0.0);
    float total_r = frag_radius + abs(frag_amp);
    if (total_r < 0.001) total_r = 0.001;
    
    float norm_dist = dist * total_r;
    float d = norm_dist - shape_r;
    
    float alpha = 1.0 - smoothstep(-0.02 * total_r, 0.02 * total_r, d);
    
    if (alpha < 0.01) discard;
    
    // Color based on wave phase
    float wave_val = cos(frag_freq * local_theta);
    vec3 col = mix(vec3(0.2, 0.3, 0.8), vec3(0.8, 0.3, 0.2), wave_val * 0.5 + 0.5);
    
    // Show slope direction
    float slope = wave_slope(local_theta, frag_freq, frag_amp, 0.0);
    col += vec3(0.0, abs(slope) * 2.0, 0.0);
    
    return vec4(col, alpha);
}

// Energy-only visualization (thermal view)
vec4 render_energy_only(float dist, float theta, float local_theta) {
    float total_r = frag_radius + abs(frag_amp);
    if (total_r < 0.001) total_r = 0.001;
    
    float norm_dist = dist * total_r;
    float alpha = 1.0 - smoothstep(0.8 * frag_radius, 1.2 * frag_radius, norm_dist);
    
    if (alpha < 0.01) discard;
    
    // Thermal colormap based on energy
    float e = clamp(frag_energy, 0.0, 3.0) / 3.0;
    
    vec3 col;
    if (e < 0.25) {
        col = mix(vec3(0.0, 0.0, 0.2), vec3(0.0, 0.0, 1.0), e * 4.0);
    } else if (e < 0.5) {
        col = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), (e - 0.25) * 4.0);
    } else if (e < 0.75) {
        col = mix(vec3(0.0, 1.0, 1.0), vec3(1.0, 1.0, 0.0), (e - 0.5) * 4.0);
    } else {
        col = mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.2, 0.0), (e - 0.75) * 4.0);
    }
    
    // Add glow
    float glow = exp(-norm_dist * 2.0 / frag_radius) * e;
    col += col * glow;
    
    return vec4(col, alpha);
}

// Minimal mode (fast, for high particle counts)
vec4 render_minimal(float dist, float theta) {
    float r = frag_radius;
    float d = dist - r * 0.8;
    float alpha = 1.0 - smoothstep(-0.1 * r, 0.1 * r, d);
    
    if (alpha < 0.01) discard;
    
    vec3 col = frag_color * (0.5 + frag_energy * 0.25);
    return vec4(col, alpha);
}

// ============================================================================
// MAIN
// ============================================================================

void main() {
    // Distance from center in UV space
    float dist = length(uv);
    
    // Early discard for pixels definitely outside
    float max_r = frag_radius + abs(frag_amp);
    if (dist > 1.2) discard;
    
    // Calculate angles
    float theta = fast_atan2(uv.y, uv.x);
    float local_theta = theta - frag_angle;
    
    // Select render mode
    vec4 result;
    
    if (render_mode == 0) {
        result = render_standard(dist, theta, local_theta);
    } else if (render_mode == 1) {
        result = render_wave_only(dist, theta, local_theta);
    } else if (render_mode == 2) {
        result = render_energy_only(dist, theta, local_theta);
    } else {
        result = render_minimal(dist, theta);
    }
    
    out_color = result;
}