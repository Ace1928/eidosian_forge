from OpenGL import platform as _p, arrays
from OpenGL.raw.GL import _types as _cs
from OpenGL.raw.GL._types import *
from OpenGL.raw.GL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLuint, _cs.GLuint64EXT, _cs.GLuint, _cs.GLuint, _cs.GLenum, _cs.GLenum, _cs.GLuint, _cs.GLenum, _cs.GLuint, _cs.GLenum, _cs.GLuint, _cs.GLenum, _cs.GLuint)
def glPresentFrameDualFillNV(video_slot, minPresentTime, beginPresentTimeId, presentDurationId, type, target0, fill0, target1, fill1, target2, fill2, target3, fill3):
    pass