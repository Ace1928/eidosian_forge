from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.EGL import _types, _glgets
from OpenGL.raw.EGL.KHR.swap_buffers_with_damage import *
from OpenGL.raw.EGL.KHR.swap_buffers_with_damage import _EXTENSION_NAME
def glInitSwapBuffersWithDamageKHR():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)