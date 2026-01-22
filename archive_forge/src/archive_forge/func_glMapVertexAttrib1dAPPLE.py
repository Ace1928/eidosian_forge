from OpenGL import platform as _p, arrays
from OpenGL.raw.GL import _types as _cs
from OpenGL.raw.GL._types import *
from OpenGL.raw.GL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLuint, _cs.GLuint, _cs.GLdouble, _cs.GLdouble, _cs.GLint, _cs.GLint, arrays.GLdoubleArray)
def glMapVertexAttrib1dAPPLE(index, size, u1, u2, stride, order, points):
    pass