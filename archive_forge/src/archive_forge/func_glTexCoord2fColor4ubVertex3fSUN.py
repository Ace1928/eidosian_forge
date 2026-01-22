from OpenGL import platform as _p, arrays
from OpenGL.raw.GL import _types as _cs
from OpenGL.raw.GL._types import *
from OpenGL.raw.GL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLfloat, _cs.GLfloat, _cs.GLubyte, _cs.GLubyte, _cs.GLubyte, _cs.GLubyte, _cs.GLfloat, _cs.GLfloat, _cs.GLfloat)
def glTexCoord2fColor4ubVertex3fSUN(s, t, r, g, b, a, x, y, z):
    pass