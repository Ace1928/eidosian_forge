from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.WGL import _types, _glgets
from OpenGL.raw.WGL.EXT.create_context_es2_profile import *
from OpenGL.raw.WGL.EXT.create_context_es2_profile import _EXTENSION_NAME
def glInitCreateContextEs2ProfileEXT():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)