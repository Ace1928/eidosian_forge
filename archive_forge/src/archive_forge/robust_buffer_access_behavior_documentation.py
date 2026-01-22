from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.KHR.robust_buffer_access_behavior import *
from OpenGL.raw.GLES2.KHR.robust_buffer_access_behavior import _EXTENSION_NAME
Return boolean indicating whether this extension is available