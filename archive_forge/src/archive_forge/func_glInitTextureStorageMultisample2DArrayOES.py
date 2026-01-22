from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.OES.texture_storage_multisample_2d_array import *
from OpenGL.raw.GLES2.OES.texture_storage_multisample_2d_array import _EXTENSION_NAME
def glInitTextureStorageMultisample2DArrayOES():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)