from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.EGL import _types, _glgets
from OpenGL.raw.EGL.KHR.stream_producer_aldatalocator import *
from OpenGL.raw.EGL.KHR.stream_producer_aldatalocator import _EXTENSION_NAME
def glInitStreamProducerAldatalocatorKHR():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)