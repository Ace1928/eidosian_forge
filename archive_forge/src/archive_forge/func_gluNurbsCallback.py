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
def gluNurbsCallback(nurb, which, CallBackFunc):
    """Dispatch to the nurb's addCallback operation"""
    return nurb.addCallback(which, CallBackFunc)