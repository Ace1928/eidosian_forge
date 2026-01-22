from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.EGL import _types, _glgets
from OpenGL.raw.EGL.MESA.platform_gbm import *
from OpenGL.raw.EGL.MESA.platform_gbm import _EXTENSION_NAME
def glInitPlatformGbmMESA():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)