from OpenGL.raw import GLU as _simple
from OpenGL.raw.GL.VERSION import GL_1_1
from OpenGL.platform import createBaseFunction
from OpenGL.GLU import glustruct
from OpenGL import arrays, wrapper
from OpenGL.platform import PLATFORM
from OpenGL.lazywrapper import lazy as _lazy
import ctypes
@_lazy(_simple.gluGetTessProperty)
def gluGetTessProperty(baseFunction, tess, which, data=None):
    """Retrieve single double for a tessellator property"""
    if data is None:
        data = _simple.GLdouble(0.0)
        baseFunction(tess, which, data)
        return data.value
    else:
        return baseFunction(tess, which, data)