from fontTools.misc.bezierTools import splitCubicAtTC
from collections import namedtuple
import math
from typing import (
@cython.locals(cost=cython.int, is_complex=cython.int)
def quadratic_to_curves(quads: List[List[Point]], max_err: float=0.5, all_cubic: bool=False) -> List[Tuple[Point, ...]]:
    """Converts a connecting list of quadratic splines to a list of quadratic
    and cubic curves.

    A quadratic spline is specified as a list of points.  Either each point is
    a 2-tuple of X,Y coordinates, or each point is a complex number with
    real/imaginary components representing X,Y coordinates.

    The first and last points are on-curve points and the rest are off-curve
    points, with an implied on-curve point in the middle between every two
    consequtive off-curve points.

    Returns:
        The output is a list of tuples of points. Points are represented
        in the same format as the input, either as 2-tuples or complex numbers.

        Each tuple is either of length three, for a quadratic curve, or four,
        for a cubic curve.  Each curve's last point is the same as the next
        curve's first point.

    Args:
        quads: quadratic splines

        max_err: absolute error tolerance; defaults to 0.5

        all_cubic: if True, only cubic curves are generated; defaults to False
    """
    is_complex = type(quads[0][0]) is complex
    if not is_complex:
        quads = [[complex(x, y) for x, y in p] for p in quads]
    q = [quads[0][0]]
    costs = [1]
    cost = 1
    for p in quads:
        assert q[-1] == p[0]
        for i in range(len(p) - 2):
            cost += 1
            costs.append(cost)
            costs.append(cost)
        qq = add_implicit_on_curves(p)[1:]
        costs.pop()
        q.extend(qq)
        cost += 1
        costs.append(cost)
    curves = spline_to_curves(q, costs, max_err, all_cubic)
    if not is_complex:
        curves = [tuple(((c.real, c.imag) for c in curve)) for curve in curves]
    return curves