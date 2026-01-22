from OpenGL import platform as _p, arrays
from OpenGL.raw.GLES2 import _types as _cs
from OpenGL.raw.GLES2._types import *
from OpenGL.raw.GLES2 import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLsizei, arrays.GLuintArray, _cs.GLenum, ctypes.c_void_p, _cs.GLsizei)
def glShaderBinary(count, shaders, binaryformat, binary, length):
    pass