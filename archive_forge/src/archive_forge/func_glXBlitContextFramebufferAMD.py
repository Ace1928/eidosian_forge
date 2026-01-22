from OpenGL import platform as _p, arrays
from OpenGL.raw.GLX import _types as _cs
from OpenGL.raw.GLX._types import *
from OpenGL.raw.GLX import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLXContext, _cs.GLint, _cs.GLint, _cs.GLint, _cs.GLint, _cs.GLint, _cs.GLint, _cs.GLint, _cs.GLint, _cs.GLbitfield, _cs.GLenum)
def glXBlitContextFramebufferAMD(dstCtx, srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1, mask, filter):
    pass