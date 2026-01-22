from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.EXT.shader_non_constant_global_initializers import *
from OpenGL.raw.GLES2.EXT.shader_non_constant_global_initializers import _EXTENSION_NAME
def glInitShaderNonConstantGlobalInitializersEXT():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)