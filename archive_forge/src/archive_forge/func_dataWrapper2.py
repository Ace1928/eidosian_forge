from OpenGL.raw import GLU as _simple
from OpenGL.raw.GL.VERSION import GL_1_1
from OpenGL.platform import createBaseFunction
from OpenGL.GLU import glustruct
from OpenGL import arrays, wrapper
from OpenGL.platform import PLATFORM
from OpenGL.lazywrapper import lazy as _lazy
import ctypes
def dataWrapper2(self, function):
    """Wrap a function which has two data-pointers as last args"""
    if function is not None and (not hasattr(function, '__call__')):
        raise TypeError('Require a callable callback, got:  %s' % (function,))

    def wrap(*args):
        """Just return the original object for polygon_data"""
        args = args[:-2] + (self.originalObject(args[-2]), self.originalObject(args[-1]))
        try:
            return function(*args)
        except Exception as err:
            err.args += (function, args)
            raise
    return wrap