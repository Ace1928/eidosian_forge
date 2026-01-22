from OpenGL.EGL import *
import itertools
def eglErrorName(value):
    """Returns error constant if known, otherwise returns value"""
    return KNOWN_ERRORS.get(value, value)