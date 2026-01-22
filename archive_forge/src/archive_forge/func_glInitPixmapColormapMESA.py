from OpenGL import platform, constant, arrays
from OpenGL import extensions, wrapper
import ctypes
from OpenGL.raw.GLX import _types, _glgets
from OpenGL.raw.GLX.MESA.pixmap_colormap import *
from OpenGL.raw.GLX.MESA.pixmap_colormap import _EXTENSION_NAME
def glInitPixmapColormapMESA():
    """Return boolean indicating whether this extension is available"""
    from OpenGL import extensions
    return extensions.hasGLExtension(_EXTENSION_NAME)