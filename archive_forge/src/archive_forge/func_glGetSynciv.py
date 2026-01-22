from OpenGL import platform as _p, arrays
from OpenGL.raw.GLES3 import _types as _cs
from OpenGL.raw.GLES3._types import *
from OpenGL.raw.GLES3 import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLsync, _cs.GLenum, _cs.GLsizei, arrays.GLsizeiArray, arrays.GLintArray)
def glGetSynciv(sync, pname, bufSize, length, values):
    pass