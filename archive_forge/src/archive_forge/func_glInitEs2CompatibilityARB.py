from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.ARB.ES2_compatibility import *
from OpenGL.raw.GL.ARB.ES2_compatibility import _EXTENSION_NAME
from OpenGL import lazywrapper as _lazywrapper
from OpenGL.arrays import GLintArray
def glInitEs2CompatibilityARB():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)