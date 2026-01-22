from OpenGL import platform as _p, arrays
from OpenGL.raw.GLES2 import _types as _cs
from OpenGL.raw.GLES2._types import *
from OpenGL.raw.GLES2 import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(_cs.GLenum, _cs.GLuint, _cs.GLenum, ctypes.c_void_p, _cs.GLbitfield, _cs.GLuint, _cs.GLsizei, _cs.GLuint, _cs.GLfloat)
def glPathGlyphIndexArrayNV(firstPathName, fontTarget, fontName, fontStyle, firstGlyphIndex, numGlyphs, pathParameterTemplate, emScale):
    pass