from OpenGL import platform as _p, arrays
from OpenGL.raw.GLES3 import _types as _cs
from OpenGL.raw.GLES3._types import *
from OpenGL.raw.GLES3 import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLuint, _cs.GLint, _cs.GLsizei, _cs.GLboolean, arrays.GLfloatArray)
def glProgramUniformMatrix4x2fv(program, location, count, transpose, value):
    pass