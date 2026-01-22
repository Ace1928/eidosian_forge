from OpenGL import arrays
from OpenGL.raw.GL._types import GLenum,GLboolean,GLsizei,GLint,GLuint
from OpenGL.raw.osmesa._types import *
from OpenGL.constant import Constant as _C
from OpenGL import platform as _p
import ctypes
Enable/disable Gallium post-process filters.

    This should be called after a context is created, but before it is
    made current for the first time.  After a context has been made
    current, this function has no effect.

    If the enable_value param is zero, the filter is disabled.  Otherwise
    the filter is enabled, and the value may control the filter's quality.
    New in Mesa 10.0
    