from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.EXT.framebuffer_object import *
from OpenGL.raw.GL.EXT.framebuffer_object import _EXTENSION_NAME
from OpenGL.lazywrapper import lazy as _lazy 
def glInitFramebufferObjectEXT():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)