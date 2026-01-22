from OpenGL import platform as _p, arrays
from OpenGL.raw.GLES1 import _types as _cs
from OpenGL.raw.GLES1._types import *
from OpenGL.raw.GLES1 import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLenum, _cs.GLfixed, _cs.GLfixed, _cs.GLint, _cs.GLint, _cs.GLfixed)
def glMap1xOES(target, u1, u2, stride, order, points):
    pass