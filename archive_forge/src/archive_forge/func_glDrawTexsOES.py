from OpenGL import platform as _p, arrays
from OpenGL.raw.GLES1 import _types as _cs
from OpenGL.raw.GLES1._types import *
from OpenGL.raw.GLES1 import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLshort, _cs.GLshort, _cs.GLshort, _cs.GLshort, _cs.GLshort)
def glDrawTexsOES(x, y, z, width, height):
    pass