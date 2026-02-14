#version 450 core

in vec3 Normal; // Normal vector passed from vertex shader
in vec2 TexCoord; // Texture coordinates passed from vertex shader

out vec4 FragColor; // Output color of the pixel

uniform sampler2D texture_diffuse; // Diffuse texture
uniform vec3 lightPos; // Light position
uniform vec3 viewPos; // Camera position

void main()
{
    // Ambient lighting
    float ambientStrength = 0.1;
    vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);

    // Diffuse lighting
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - vec3(gl_FragCoord));
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);

    // Specular lighting
    float specularStrength = 0.5;
    vec3 viewDir = normalize(viewPos - vec3(gl_FragCoord));
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);

    vec3 result = (ambient + diffuse + specular) * texture(texture_diffuse, TexCoord).rgb;
    FragColor = vec4(result, 1.0);
}