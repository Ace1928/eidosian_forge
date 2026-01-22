from fontTools.misc.bezierTools import splitCubicAtTC
from collections import namedtuple
import math
from typing import (
@cython.cfunc
@cython.locals(start=cython.int, n=cython.int, k=cython.int, prod_ratio=cython.double, sum_ratio=cython.double, ratio=cython.double, t=cython.double, p0=cython.complex, p1=cython.complex, p2=cython.complex, p3=cython.complex)
def merge_curves(curves, start, n):
    """Give a cubic-Bezier spline, reconstruct one cubic-Bezier
    that has the same endpoints and tangents and approxmates
    the spline."""
    prod_ratio = 1.0
    sum_ratio = 1.0
    ts = [1]
    for k in range(1, n):
        ck = curves[start + k]
        c_before = curves[start + k - 1]
        assert ck[0] == c_before[3]
        ratio = abs(ck[1] - ck[0]) / abs(c_before[3] - c_before[2])
        prod_ratio *= ratio
        sum_ratio += prod_ratio
        ts.append(sum_ratio)
    ts = [t / sum_ratio for t in ts[:-1]]
    p0 = curves[start][0]
    p1 = curves[start][1]
    p2 = curves[start + n - 1][2]
    p3 = curves[start + n - 1][3]
    p1 = p0 + (p1 - p0) / (ts[0] if ts else 1)
    p2 = p3 + (p2 - p3) / (1 - ts[-1] if ts else 1)
    curve = (p0, p1, p2, p3)
    return (curve, ts)