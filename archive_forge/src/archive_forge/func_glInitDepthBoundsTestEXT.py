from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.EXT.depth_bounds_test import *
from OpenGL.raw.GL.EXT.depth_bounds_test import _EXTENSION_NAME
def glInitDepthBoundsTestEXT():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)