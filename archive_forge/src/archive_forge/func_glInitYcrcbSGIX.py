from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.SGIX.ycrcb import *
from OpenGL.raw.GL.SGIX.ycrcb import _EXTENSION_NAME
from OpenGL import images as _i
def glInitYcrcbSGIX():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)