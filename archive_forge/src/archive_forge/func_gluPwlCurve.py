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
@_lazy(_simple.gluPwlCurve)
def gluPwlCurve(baseFunction, nurb, data, type):
    """gluPwlCurve -- piece-wise linear curve within GLU context

    data -- the data-array
    type -- determines number of elements/data-point
    """
    data = arrays.GLfloatArray.asArray(data)
    if type == _simple.GLU_MAP1_TRIM_2:
        divisor = 2
    elif type == _simple.GLU_MAP_TRIM_3:
        divisor = 3
    else:
        raise ValueError('Unrecognised type constant: %s' % type)
    size = arrays.GLfloatArray.arraySize(data)
    size = int(size // divisor)
    return baseFunction(nurb, size, data, divisor, type)