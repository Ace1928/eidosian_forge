from OpenGL import platform as _p, arrays
from OpenGL.raw.GL import _types as _cs
from OpenGL.raw.GL._types import *
from OpenGL.raw.GL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLuint, _cs.GLenum, _cs.GLint, _cs.GLenum, _cs.GLsizei, _cs.GLsizei, _cs.GLint, _cs.GLsizei, ctypes.c_void_p)
def glCompressedTextureImage2DEXT(texture, target, level, internalformat, width, height, border, imageSize, bits):
    pass