from OpenGL import constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.EGL import _types, _glgets
from OpenGL.raw.EGL.EXT.platform_base import *
from OpenGL.raw.EGL.EXT.platform_base import _EXTENSION_NAME
def glInitPlatformBaseEXT():
    """Return boolean indicating whether this extension is available"""
    return extensions.hasGLExtension(_EXTENSION_NAME)