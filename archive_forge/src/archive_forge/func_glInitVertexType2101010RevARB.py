from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.ARB.vertex_type_2_10_10_10_rev import *
from OpenGL.raw.GL.ARB.vertex_type_2_10_10_10_rev import _EXTENSION_NAME
def glInitVertexType2101010RevARB():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)