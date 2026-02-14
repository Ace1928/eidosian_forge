#version 450 core

layout(location = 0) in vec3 aPos;   // Position of the vertex
layout(location = 1) in vec3 aNormal; // Normal vector
layout(location = 2) in vec2 aTexCoord; // Texture coordinates

uniform mat4 model; // Model matrix
uniform mat4 view;  // View matrix
uniform mat4 projection; // Projection matrix

out vec3 Normal; // Normal to pass to fragment shader
out vec2 TexCoord; // Texture coordinates to pass to fragment shader

void main()
{
    gl_Position = projection * view * model * vec4(aPos, 1.0);
    Normal = mat3(transpose(inverse(model))) * aNormal;
    TexCoord = aTexCoord;
}