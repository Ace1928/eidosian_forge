from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.NV.shader_atomic_float64 import *
from OpenGL.raw.GL.NV.shader_atomic_float64 import _EXTENSION_NAME
def glInitShaderAtomicFloat64NV():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)