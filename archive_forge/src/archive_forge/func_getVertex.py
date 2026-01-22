from OpenGL import contextdata
from OpenGL.GL.VERSION import GL_1_1 as _simple
def getVertex(buffer, bufferIndex):
    end = bufferIndex + size
    colorEnd = end + colorSize
    textureEnd = colorEnd + 4
    return ((buffer[bufferIndex:end], buffer[end:colorEnd], buffer[colorEnd:textureEnd]), textureEnd)