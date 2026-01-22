from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.ARB.vertex_shader import *
from OpenGL.raw.GL.ARB.vertex_shader import _EXTENSION_NAME
from OpenGL._bytes import bytes, _NULL_8_BYTE, as_8_bit
from OpenGL.lazywrapper import lazy as _lazy
from OpenGL.GL.ARB.shader_objects import glGetObjectParameterivARB
def glInitVertexShaderARB():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)