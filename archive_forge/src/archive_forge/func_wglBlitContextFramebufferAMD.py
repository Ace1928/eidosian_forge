from OpenGL import platform as _p, arrays
from OpenGL.raw.WGL import _types as _cs
from OpenGL.raw.WGL._types import *
from OpenGL.raw.WGL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(_cs.VOID, _cs.HGLRC, _cs.GLint, _cs.GLint, _cs.GLint, _cs.GLint, _cs.GLint, _cs.GLint, _cs.GLint, _cs.GLint, _cs.GLbitfield, _cs.GLenum)
def wglBlitContextFramebufferAMD(dstCtx, srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter):
    pass