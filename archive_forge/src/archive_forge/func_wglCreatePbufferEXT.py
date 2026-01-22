from OpenGL import platform as _p, arrays
from OpenGL.raw.WGL import _types as _cs
from OpenGL.raw.WGL._types import *
from OpenGL.raw.WGL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(_cs.HPBUFFEREXT, _cs.HDC, _cs.c_int, _cs.c_int, _cs.c_int, ctypes.POINTER(_cs.c_int))
def wglCreatePbufferEXT(hDC, iPixelFormat, iWidth, iHeight, piAttribList):
    pass