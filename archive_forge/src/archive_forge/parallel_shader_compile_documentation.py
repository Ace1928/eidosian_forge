from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.KHR.parallel_shader_compile import *
from OpenGL.raw.GLES2.KHR.parallel_shader_compile import _EXTENSION_NAME
Return boolean indicating whether this extension is available