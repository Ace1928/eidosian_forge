from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.EXT.gpu_program_parameters import *
from OpenGL.raw.GL.EXT.gpu_program_parameters import _EXTENSION_NAME
def glInitGpuProgramParametersEXT():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)