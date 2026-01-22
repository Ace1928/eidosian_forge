from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GL import _types, _glgets
from OpenGL.raw.GL.NV.gpu_program5_mem_extended import *
from OpenGL.raw.GL.NV.gpu_program5_mem_extended import _EXTENSION_NAME
def glInitGpuProgram5MemExtendedNV():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)