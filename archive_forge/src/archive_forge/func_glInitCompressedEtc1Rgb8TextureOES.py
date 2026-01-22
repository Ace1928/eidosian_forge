from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.OES.compressed_ETC1_RGB8_texture import *
from OpenGL.raw.GLES2.OES.compressed_ETC1_RGB8_texture import _EXTENSION_NAME
def glInitCompressedEtc1Rgb8TextureOES():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)