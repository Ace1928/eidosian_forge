from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLX import _types, _glgets
from OpenGL.raw.GLX.INTEL.swap_event import *
from OpenGL.raw.GLX.INTEL.swap_event import _EXTENSION_NAME
def glInitSwapEventINTEL():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)