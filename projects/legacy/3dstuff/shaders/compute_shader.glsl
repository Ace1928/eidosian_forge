#version 450 core

layout (local_size_x = 1024) in;

struct Particle {
    vec3 position;
    vec3 velocity;
};

layout(std430, binding = 0) buffer Particles {
    Particle particles[];
};

uniform float deltaTime;

void main()
{
    uint idx = gl_GlobalInvocationID.x;
    particles[idx].position += particles[idx].velocity * deltaTime;
}