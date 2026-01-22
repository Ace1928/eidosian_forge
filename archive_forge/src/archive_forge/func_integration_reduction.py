from functools import cmp_to_key
from sympy.abc import x, y, z
from sympy.core import S, diff, Expr, Symbol
from sympy.core.sympify import _sympify
from sympy.geometry import Segment2D, Polygon, Point, Point2D
from sympy.polys.polytools import LC, gcd_list, degree_list, Poly
from sympy.simplify.simplify import nsimplify
def integration_reduction(facets, index, a, b, expr, dims, degree):
    """Helper method for main_integrate. Returns the value of the input
    expression evaluated over the polytope facet referenced by a given index.

    Parameters
    ===========

    facets :
        List of facets of the polytope.
    index :
        Index referencing the facet to integrate the expression over.
    a :
        Hyperplane parameter denoting direction.
    b :
        Hyperplane parameter denoting distance.
    expr :
        The expression to integrate over the facet.
    dims :
        List of symbols denoting axes.
    degree :
        Degree of the homogeneous polynomial.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.integrals.intpoly import integration_reduction,    hyperplane_parameters
    >>> from sympy import Point, Polygon
    >>> triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    >>> facets = triangle.sides
    >>> a, b = hyperplane_parameters(triangle)[0]
    >>> integration_reduction(facets, 0, a, b, 1, (x, y), 0)
    5
    """
    expr = _sympify(expr)
    if expr.is_zero:
        return expr
    value = S.Zero
    x0 = facets[index].points[0]
    m = len(facets)
    gens = (x, y)
    inner_product = diff(expr, gens[0]) * x0[0] + diff(expr, gens[1]) * x0[1]
    if inner_product != 0:
        value += integration_reduction(facets, index, a, b, inner_product, dims, degree - 1)
    value += left_integral2D(m, index, facets, x0, expr, gens)
    return value / (len(dims) + degree - 1)