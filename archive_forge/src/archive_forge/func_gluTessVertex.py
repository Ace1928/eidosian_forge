from OpenGL.raw import GLU as _simple
from OpenGL.raw.GL.VERSION import GL_1_1
from OpenGL.platform import createBaseFunction
from OpenGL.GLU import glustruct
from OpenGL import arrays, wrapper
from OpenGL.platform import PLATFORM
from OpenGL.lazywrapper import lazy as _lazy
import ctypes
def gluTessVertex(self, location, data=None):
    """Add a vertex to this tessellator, storing data for later lookup"""
    vertexCache = getattr(self, 'vertexCache', None)
    if vertexCache is None:
        self.vertexCache = []
        vertexCache = self.vertexCache
    location = arrays.GLdoubleArray.asArray(location, GL_1_1.GL_DOUBLE)
    if arrays.GLdoubleArray.arraySize(location) != 3:
        raise ValueError('Require 3 doubles for array location, got: %s' % (location,))
    oorValue = self.noteObject(data)
    vp = ctypes.c_void_p(oorValue)
    self.vertexCache.append(location)
    return gluTessVertexBase(self, location, vp)