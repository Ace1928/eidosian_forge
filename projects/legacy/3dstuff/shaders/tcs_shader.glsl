#version 450 core

layout(vertices = 3) out;

in vec3 vPosition[];

uniform int tessellationLevel;

void main()
{
    if (gl_InvocationID == 0)
    {
        gl_TessLevelInner[0] = tessellationLevel;
        gl_TessLevelOuter[0] = tessellationLevel;
        gl_TessLevelOuter[1] = tessellationLevel;
        gl_TessLevelOuter[2] = tessellationLevel;
    }
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}