from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.EXT.multiview_tessellation_geometry_shader import *
from OpenGL.raw.GLES2.EXT.multiview_tessellation_geometry_shader import _EXTENSION_NAME
def glInitMultiviewTessellationGeometryShaderEXT():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)