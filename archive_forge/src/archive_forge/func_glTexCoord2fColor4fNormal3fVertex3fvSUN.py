from OpenGL import platform as _p, arrays
from OpenGL.raw.GL import _types as _cs
from OpenGL.raw.GL._types import *
from OpenGL.raw.GL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, arrays.GLfloatArray, arrays.GLfloatArray, arrays.GLfloatArray, arrays.GLfloatArray)
def glTexCoord2fColor4fNormal3fVertex3fvSUN(tc, c, n, v):
    pass