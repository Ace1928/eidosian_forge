from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.ARB.framebuffer_object import *
from OpenGL.raw.GL.ARB.framebuffer_object import _EXTENSION_NAME
from OpenGL.lazywrapper import lazy as _lazy 
from OpenGL import images
from OpenGL.raw.GL.VERSION.GL_1_1 import GL_UNSIGNED_INT
def glInitFramebufferObjectARB():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)