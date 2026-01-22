from OpenGL import platform as _p, arrays
from OpenGL.raw.GL import _types as _cs
from OpenGL.raw.GL._types import *
from OpenGL.raw.GL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(_cs.GLuint, _cs.GLuint, _cs.GLsizei, arrays.GLuintArray, arrays.GLuintArray, arrays.GLuintArray, arrays.GLuintArray, arrays.GLsizeiArray, arrays.GLcharArray)
def glGetDebugMessageLogARB(count, bufSize, sources, types, ids, severities, lengths, messageLog):
    pass