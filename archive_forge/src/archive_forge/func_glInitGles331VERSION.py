from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES3 import _types, _glgets
from OpenGL.raw.GLES3.VERSION.GLES3_3_1 import *
from OpenGL.raw.GLES3.VERSION.GLES3_3_1 import _EXTENSION_NAME
def glInitGles331VERSION():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)