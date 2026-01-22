from OpenGL import platform as _p, arrays
from OpenGL.raw.GL import _types as _cs
from OpenGL.raw.GL._types import *
from OpenGL.raw.GL import _errors
from OpenGL.constant import Constant as _C
import ctypes
@_f
@_p.types(None, _cs.GLuint, _cs.GLuint64EXT, _cs.GLuint, _cs.GLuint, _cs.GLenum, _cs.GLenum, _cs.GLuint, _cs.GLuint, _cs.GLenum, _cs.GLuint, _cs.GLuint)
def glPresentFrameKeyedNV(video_slot, minPresentTime, beginPresentTimeId, presentDurationId, type, target0, fill0, key0, target1, fill1, key1):
    pass