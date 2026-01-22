from OpenGL.raw import GLU as _simple
from OpenGL.raw.GL.VERSION import GL_1_1
from OpenGL.platform import createBaseFunction
from OpenGL.GLU import glustruct
from OpenGL import arrays, wrapper
from OpenGL.platform import PLATFORM
from OpenGL.lazywrapper import lazy as _lazy
import ctypes
def combineWrapper(self, function):
    """Wrap a Python function with ctypes-compatible wrapper for combine callback

        For a Python combine callback, the signature looks like this:
            def combine(
                GLdouble coords[3],
                void *vertex_data[4],
                GLfloat weight[4]
            ):
                return data
        While the C signature looks like this:
            void combine(
                GLdouble coords[3],
                void *vertex_data[4],
                GLfloat weight[4],
                void **outData
            )
        """
    if function is not None and (not hasattr(function, '__call__')):
        raise TypeError('Require a callable callback, got:  %s' % (function,))

    def wrap(coords, vertex_data, weight, outData, *args):
        """The run-time wrapper around the function"""
        coords = self.ptrAsArray(coords, 3, arrays.GLdoubleArray)
        weight = self.ptrAsArray(weight, 4, arrays.GLfloatArray)
        vertex_data = [self.originalObject(vertex_data[i]) for i in range(4)]
        args = tuple([self.originalObject(x) for x in args])
        try:
            result = function(coords, vertex_data, weight, *args)
        except Exception as err:
            raise err.__class__('Failure during combine callback %r with args( %s,%s,%s,*%s):\n%s' % (function, coords, vertex_data, weight, args, str(err)))
        outP = ctypes.c_void_p(self.noteObject(result))
        if outData:
            outData[0] = outP
        else:
            raise RuntimeError('Null outData passed to callback')
        return None
    return wrap