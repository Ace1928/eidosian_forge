#version 330 core

in vec2 uv;
in vec3 color;
in float radius;
in float freq;
in float amp;
in float angle;
in float energy;

out vec4 fragColor;

void main() {
    float dist = length(uv); 
    float scale_factor = 2.5;
    float r_norm = dist * scale_factor * (radius + amp);
    
    float theta = atan(uv.y, uv.x); 
    float local_theta = theta - angle;
    
    float shape_r = radius + amp * cos(freq * local_theta);
    float d = r_norm - shape_r;
    
    float alpha = 1.0 - smoothstep(-1.0, 1.0, d);
    
    if (alpha <= 0.0) discard;
    
    // Energy Visualization
    // Low energy -> Dark / Desaturated
    // High energy -> Bright / Blooming
    float e_factor = clamp(energy, 0.0, 2.0);
    
    vec3 col = color * e_factor;
    
    // Core glow based on energy
    float glow = exp(-d * 0.1) * e_factor;
    col += vec3(glow * 0.3);
    
    float perimeter = 1.0 - smoothstep(0.0, 2.0, abs(d));
    col += vec3(1.0) * perimeter * 0.8 * e_factor;
    
    fragColor = vec4(col, alpha);
}