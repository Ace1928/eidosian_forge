from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.VERSION.GL_2_0 import *
from OpenGL.raw.GL.VERSION.GL_2_0 import _EXTENSION_NAME
import OpenGL
from OpenGL import _configflags
from OpenGL._bytes import bytes, _NULL_8_BYTE, as_8_bit
from OpenGL.raw.GL.ARB.shader_objects import GL_OBJECT_COMPILE_STATUS_ARB as GL_OBJECT_COMPILE_STATUS
from OpenGL.raw.GL.ARB.shader_objects import GL_OBJECT_LINK_STATUS_ARB as GL_OBJECT_LINK_STATUS
from OpenGL.raw.GL.ARB.shader_objects import GL_OBJECT_ACTIVE_UNIFORMS_ARB as GL_OBJECT_ACTIVE_UNIFORMS
from OpenGL.raw.GL.ARB.shader_objects import GL_OBJECT_ACTIVE_UNIFORM_MAX_LENGTH_ARB as GL_OBJECT_ACTIVE_UNIFORM_MAX_LENGTH
from OpenGL.lazywrapper import lazy as _lazy
from OpenGL.raw.GL import _errors
from OpenGL import converters, error, contextdata
from OpenGL.arrays.arraydatatype import ArrayDatatype, GLenumArray
def glInitGl20VERSION():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)