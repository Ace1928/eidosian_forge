from OpenGL.raw import GLU as _simple
from OpenGL.raw.GL.VERSION import GL_1_1
from OpenGL.platform import createBaseFunction
from OpenGL.GLU import glustruct
from OpenGL import arrays, wrapper
from OpenGL.platform import PLATFORM
from OpenGL.lazywrapper import lazy as _lazy
import ctypes
def gluTessBeginPolygon(self, data):
    """Note the object pointer to return it as a Python object"""
    return _simple.gluTessBeginPolygon(self, ctypes.c_void_p(self.noteObject(data)))