from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES1 import _types, _glgets
from OpenGL.raw.GLES1.APPLE.texture_2D_limited_npot import *
from OpenGL.raw.GLES1.APPLE.texture_2D_limited_npot import _EXTENSION_NAME
def glInitTexture2DLimitedNpotAPPLE():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)