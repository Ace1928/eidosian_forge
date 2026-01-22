import logging
from OpenGL.GLES2 import *
from OpenGL._bytes import bytes,unicode,as_8_bit
def compileShader(source, shaderType):
    """Compile shader source of given type

    source -- GLSL source-code for the shader
    shaderType -- GLenum GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, etc,

    returns GLuint compiled shader reference
    raises RuntimeError when a compilation failure occurs
    """
    if isinstance(source, (bytes, unicode)):
        source = [source]
    source = [as_8_bit(s) for s in source]
    shader = glCreateShader(shaderType)
    glShaderSource(shader, source)
    glCompileShader(shader)
    result = glGetShaderiv(shader, GL_COMPILE_STATUS)
    if result == GL_FALSE:
        raise RuntimeError('Shader compile failure (%s): %s' % (result, glGetShaderInfoLog(shader)), source, shaderType)
    return shader