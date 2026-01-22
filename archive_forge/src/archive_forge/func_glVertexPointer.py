from OpenGL import platform as _p, arrays
from OpenGL.raw.GLES1 import _types as _cs
from OpenGL.raw.GLES1._types import *
from OpenGL.raw.GLES1 import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLint, _cs.GLenum, _cs.GLsizei, ctypes.c_void_p)
def glVertexPointer(size, type, stride, pointer):
    pass