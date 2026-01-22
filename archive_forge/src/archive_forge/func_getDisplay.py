import ctypes
from OpenGL._opaque import opaque_pointer_cls as _opaque_pointer_cls
from OpenGL import platform as _p
from OpenGL import extensions 
from OpenGL._bytes import as_8_bit
def getDisplay(self):
    """Retrieve the currently-bound, or the default, display"""
    from OpenGL.EGL import eglGetCurrentDisplay, eglGetDisplay, EGL_DEFAULT_DISPLAY
    return eglGetCurrentDisplay() or eglGetDisplay(EGL_DEFAULT_DISPLAY)