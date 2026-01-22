from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.ATI.draw_buffers import *
from OpenGL.raw.GL.ATI.draw_buffers import _EXTENSION_NAME
from OpenGL.lazywrapper import lazy as _lazy
def glInitDrawBuffersATI():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)