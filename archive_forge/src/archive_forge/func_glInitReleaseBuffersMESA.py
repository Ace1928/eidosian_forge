from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLX import _types, _glgets
from OpenGL.raw.GLX.MESA.release_buffers import *
from OpenGL.raw.GLX.MESA.release_buffers import _EXTENSION_NAME
def glInitReleaseBuffersMESA():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)