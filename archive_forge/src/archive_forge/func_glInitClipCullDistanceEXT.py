from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLES2 import _types, _glgets
from OpenGL.raw.GLES2.EXT.clip_cull_distance import *
from OpenGL.raw.GLES2.EXT.clip_cull_distance import _EXTENSION_NAME
def glInitClipCullDistanceEXT():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)