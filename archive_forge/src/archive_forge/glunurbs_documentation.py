from OpenGL.raw import GLU as _simple
from OpenGL import platform, converters, wrapper
from OpenGL.GLU import glustruct
from OpenGL.lazywrapper import lazy as _lazy
from OpenGL import arrays, error
import ctypes
import weakref
from OpenGL.platform import PLATFORM
import OpenGL
from OpenGL import _configflags
Texture coordinate callback

        NOTE: there is no way for *us* to tell what size the array is, you will
        get back a raw data-point, not an array, as you do for all other callback
        types!!!
        