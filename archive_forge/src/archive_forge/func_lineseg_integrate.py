from functools import cmp_to_key
from sympy.abc import x, y, z
from sympy.core import S, diff, Expr, Symbol
from sympy.core.sympify import _sympify
from sympy.geometry import Segment2D, Polygon, Point, Point2D
from sympy.polys.polytools import LC, gcd_list, degree_list, Poly
from sympy.simplify.simplify import nsimplify
def lineseg_integrate(polygon, index, line_seg, expr, degree):
    """Helper function to compute the line integral of ``expr`` over ``line_seg``.

    Parameters
    ===========

    polygon :
        Face of a 3-Polytope.
    index :
        Index of line_seg in polygon.
    line_seg :
        Line Segment.

    Examples
    ========

    >>> from sympy.integrals.intpoly import lineseg_integrate
    >>> polygon = [(0, 5, 0), (5, 5, 0), (5, 5, 5), (0, 5, 5)]
    >>> line_seg = [(0, 5, 0), (5, 5, 0)]
    >>> lineseg_integrate(polygon, 0, line_seg, 1, 0)
    5
    """
    expr = _sympify(expr)
    if expr.is_zero:
        return S.Zero
    result = S.Zero
    x0 = line_seg[0]
    distance = norm(tuple([line_seg[1][i] - line_seg[0][i] for i in range(3)]))
    if isinstance(expr, Expr):
        expr_dict = {x: line_seg[1][0], y: line_seg[1][1], z: line_seg[1][2]}
        result += distance * expr.subs(expr_dict)
    else:
        result += distance * expr
    expr = diff(expr, x) * x0[0] + diff(expr, y) * x0[1] + diff(expr, z) * x0[2]
    result += lineseg_integrate(polygon, index, line_seg, expr, degree - 1)
    result /= degree + 1
    return result