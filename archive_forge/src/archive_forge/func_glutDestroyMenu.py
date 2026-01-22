from OpenGL.platform import CurrentContextIsValid, GLUT_GUARD_CALLBACKS, PLATFORM
from OpenGL import contextdata, error, platform, logs
from OpenGL.raw import GLUT as _simple
from OpenGL._bytes import bytes, unicode,as_8_bit
import ctypes, os, sys, traceback
from OpenGL._bytes import long, integer_types
def glutDestroyMenu(cls, menu):
    """Destroy (cleanup) the given menu
        
        Deregister's the interal pointer to the menu callback 
        
        returns None
        """
    result = _simple.glutDestroyMenu(menu)
    contextdata.delValue(('menucallback', menu))
    return result