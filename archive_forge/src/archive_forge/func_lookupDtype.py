import operator
from OpenGL.arrays import buffers
from OpenGL.raw.GL import _types 
from OpenGL.raw.GL.VERSION import GL_1_1
from OpenGL import constant, error
def lookupDtype(char):
    return numpy.zeros((1,), dtype=char).dtype