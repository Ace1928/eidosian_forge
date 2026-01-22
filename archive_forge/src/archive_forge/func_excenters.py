from sympy.core import Expr, S, oo, pi, sympify
from sympy.core.evalf import N
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import _symbol, Dummy, Symbol
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import cos, sin, tan
from .ellipse import Circle
from .entity import GeometryEntity, GeometrySet
from .exceptions import GeometryError
from .line import Line, Segment, Ray
from .point import Point
from sympy.logic import And
from sympy.matrices import Matrix
from sympy.simplify.simplify import simplify
from sympy.solvers.solvers import solve
from sympy.utilities.iterables import has_dups, has_variety, uniq, rotate_left, least_rotation
from sympy.utilities.misc import as_int, func_name
from mpmath.libmp.libmpf import prec_to_dps
import warnings
@property
def excenters(self):
    """Excenters of the triangle.

        An excenter is the center of a circle that is tangent to a side of the
        triangle and the extensions of the other two sides.

        Returns
        =======

        excenters : dict


        Examples
        ========

        The excenters are keyed to the side of the triangle to which their corresponding
        excircle is tangent: The center is keyed, e.g. the excenter of a circle touching
        side 0 is:

        >>> from sympy import Point, Triangle
        >>> p1, p2, p3 = Point(0, 0), Point(6, 0), Point(0, 2)
        >>> t = Triangle(p1, p2, p3)
        >>> t.excenters[t.sides[0]]
        Point2D(12*sqrt(10), 2/3 + sqrt(10)/3)

        See Also
        ========

        sympy.geometry.polygon.Triangle.exradii

        References
        ==========

        .. [1] https://mathworld.wolfram.com/Excircles.html

        """
    s = self.sides
    v = self.vertices
    a = s[0].length
    b = s[1].length
    c = s[2].length
    x = [v[0].x, v[1].x, v[2].x]
    y = [v[0].y, v[1].y, v[2].y]
    exc_coords = {'x1': simplify(-a * x[0] + b * x[1] + c * x[2] / (-a + b + c)), 'x2': simplify(a * x[0] - b * x[1] + c * x[2] / (a - b + c)), 'x3': simplify(a * x[0] + b * x[1] - c * x[2] / (a + b - c)), 'y1': simplify(-a * y[0] + b * y[1] + c * y[2] / (-a + b + c)), 'y2': simplify(a * y[0] - b * y[1] + c * y[2] / (a - b + c)), 'y3': simplify(a * y[0] + b * y[1] - c * y[2] / (a + b - c))}
    excenters = {s[0]: Point(exc_coords['x1'], exc_coords['y1']), s[1]: Point(exc_coords['x2'], exc_coords['y2']), s[2]: Point(exc_coords['x3'], exc_coords['y3'])}
    return excenters