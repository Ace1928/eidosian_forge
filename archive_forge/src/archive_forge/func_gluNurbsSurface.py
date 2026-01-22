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
@_lazy(_simple.gluNurbsSurface)
def gluNurbsSurface(baseFunction, nurb, sKnots, tKnots, control, type):
    """Pythonic version of gluNurbsSurface

    Calculates knotCount, stride, and order automatically
    """
    sKnots = arrays.GLfloatArray.asArray(sKnots)
    sKnotCount = arrays.GLfloatArray.arraySize(sKnots)
    tKnots = arrays.GLfloatArray.asArray(tKnots)
    tKnotCount = arrays.GLfloatArray.arraySize(tKnots)
    control = arrays.GLfloatArray.asArray(control)
    try:
        length, width, step = arrays.GLfloatArray.dimensions(control)
    except ValueError as err:
        raise error.GLUError('Need a 3-dimensional control array')
    sOrder = sKnotCount - length
    tOrder = tKnotCount - width
    sStride = width * step
    tStride = step
    if _configflags.ERROR_CHECKING:
        checkOrder(sOrder, sKnotCount, 'sOrder of NURBS surface')
        checkOrder(tOrder, tKnotCount, 'tOrder of NURBS surface')
        checkKnots(sKnots, 'sKnots of NURBS surface')
        checkKnots(tKnots, 'tKnots of NURBS surface')
    if not (sKnotCount - sOrder) * (tKnotCount - tOrder) == length * width:
        raise error.GLUError('Invalid NURB structure', nurb, sKnotCount, sKnots, tKnotCount, tKnots, sStride, tStride, control, sOrder, tOrder, type)
    result = baseFunction(nurb, sKnotCount, sKnots, tKnotCount, tKnots, sStride, tStride, control, sOrder, tOrder, type)
    return result