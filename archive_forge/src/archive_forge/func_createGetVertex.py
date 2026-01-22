from OpenGL import contextdata
from OpenGL.GL.VERSION import GL_1_1 as _simple
def createGetVertex():
    mode = contextdata.getValue('GL_FEEDBACK_BUFFER_TYPE')
    indexMode = _simple.glGetBooleanv(_simple.GL_INDEX_MODE)
    colorSize = [4, 1][int(indexMode)]
    if mode in (_simple.GL_2D, _simple.GL_3D):
        if mode == _simple.GL_2D:
            size = 2
        else:
            size = 3

        def getVertex(buffer, bufferIndex):
            end = bufferIndex + size
            return ((buffer[bufferIndex:end], None, None), end)
    elif mode == _simple.GL_3D_COLOR:

        def getVertex(buffer, bufferIndex):
            end = bufferIndex + 3
            colorEnd = end + colorSize
            return ((buffer[bufferIndex:end], buffer[end:colorEnd], None), colorEnd)
    else:
        if mode == _simple.GL_3D_COLOR_TEXTURE:
            size = 3
        else:
            size = 4

        def getVertex(buffer, bufferIndex):
            end = bufferIndex + size
            colorEnd = end + colorSize
            textureEnd = colorEnd + 4
            return ((buffer[bufferIndex:end], buffer[end:colorEnd], buffer[colorEnd:textureEnd]), textureEnd)
    return getVertex