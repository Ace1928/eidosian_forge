from fontTools.misc.arrayTools import calcBounds, sectRect, rectArea
from fontTools.misc.transform import Identity
import math
from collections import namedtuple
from math import sqrt, acos, cos, pi
@cython.locals(pt1=cython.complex, pt2=cython.complex, pt3=cython.complex, pt4=cython.complex, a=cython.complex, b=cython.complex, c=cython.complex, d=cython.complex)
def splitCubicAtTC(pt1, pt2, pt3, pt4, *ts):
    """Split a cubic Bezier curve at one or more values of t.

    Args:
        pt1,pt2,pt3,pt4: Control points of the Bezier as complex numbers..
        *ts: Positions at which to split the curve.

    Yields:
        Curve segments (each curve segment being four complex numbers).
    """
    a, b, c, d = calcCubicParametersC(pt1, pt2, pt3, pt4)
    yield from _splitCubicAtTC(a, b, c, d, *ts)