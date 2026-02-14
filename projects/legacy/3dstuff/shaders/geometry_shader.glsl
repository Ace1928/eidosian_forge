#version 450 core

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec3 Normal[];
in vec2 TexCoord[];

out vec3 geom_Normal;
out vec2 geom_TexCoord;

void main()
{
    for(int i = 0; i < gl_in.length(); i++)
    {
        gl_Position = gl_in[i].gl_Position;
        geom_Normal = Normal[i];
        geom_TexCoord = TexCoord[i];
        EmitVertex();
    }
    EndPrimitive();
}