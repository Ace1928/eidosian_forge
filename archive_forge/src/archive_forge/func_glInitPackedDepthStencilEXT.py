from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.EXT.packed_depth_stencil import *
from OpenGL.raw.GL.EXT.packed_depth_stencil import _EXTENSION_NAME
from OpenGL import images 
from OpenGL.raw.GL.VERSION.GL_1_1 import GL_UNSIGNED_INT
def glInitPackedDepthStencilEXT():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)