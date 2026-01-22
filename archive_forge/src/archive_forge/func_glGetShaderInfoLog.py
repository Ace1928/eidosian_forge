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
@_lazy(glGetShaderInfoLog)
def glGetShaderInfoLog(baseOperation, obj):
    """Retrieve the shader's error messages as a Python string

    returns string which is '' if no message
    """
    target = GLsizei()
    glGetShaderiv(obj, GL_INFO_LOG_LENGTH, target)
    length = target.value
    if length > 0:
        log = ctypes.create_string_buffer(length)
        baseOperation(obj, length, None, log)
        return log.value.strip(_NULL_8_BYTE)
    return ''