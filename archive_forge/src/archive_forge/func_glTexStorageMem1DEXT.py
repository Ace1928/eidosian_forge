from OpenGL import platform as _p, arrays
from OpenGL.raw.GLES2 import _types as _cs
from OpenGL.raw.GLES2._types import *
from OpenGL.raw.GLES2 import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLenum, _cs.GLsizei, _cs.GLenum, _cs.GLsizei, _cs.GLuint, _cs.GLuint64)
def glTexStorageMem1DEXT(target, levels, internalFormat, width, memory, offset):
    pass