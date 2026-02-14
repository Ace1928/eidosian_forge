#version 450 core

layout(triangles, equal_spacing, ccw) in;

in vec3 vPosition[];

uniform mat4 projectionMatrix;
uniform mat4 viewMatrix;
uniform mat4 modelMatrix;

void main()
{
    vec3 p0 = gl_TessCoord.x * vPosition[0];
    vec3 p1 = gl_TessCoord.y * vPosition[1];
    vec3 p2 = gl_TessCoord.z * vPosition[2];
    vec3 position = p0 + p1 + p2;

    gl_Position = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
}