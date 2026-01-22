from OpenGL.raw.GL.VERSION import GL_1_1 as _simple
from OpenGL import arrays
from OpenGL import error
from OpenGL import _configflags
import ctypes
def formatToComponentCount(format):
    """Given an OpenGL image format specification, get components/pixel"""
    size = COMPONENT_COUNTS.get(format)
    if size is None:
        raise ValueError('Unrecognised image format: %r' % (format,))
    return size