from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.VERSION.GLES2_2_0 import *
from OpenGL.raw.GLES2.VERSION.GLES2_2_0 import _EXTENSION_NAME
from OpenGL import converters
from OpenGL.lazywrapper import lazy as _lazy
from OpenGL._bytes import _NULL_8_BYTE
from OpenGL import contextdata 
@_lazy(glGetAttachedShaders)
def glGetAttachedShaders(baseOperation, obj):
    """Retrieve the attached objects as an array of GLhandle instances"""
    length = glGetProgramiv(obj, GL_ATTACHED_SHADERS)
    if length > 0:
        storage = arrays.GLuintArray.zeros((length,))
        baseOperation(obj, length, None, storage)
        return storage
    return arrays.GLuintArray.zeros((0,))