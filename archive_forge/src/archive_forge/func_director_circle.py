from sympy.core.expr import Expr
from sympy.core.relational import Eq
from sympy.core import S, pi, sympify
from sympy.core.evalf import N
from sympy.core.parameters import global_parameters
from sympy.core.logic import fuzzy_bool
from sympy.core.numbers import Rational, oo
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, uniquely_named_symbol, _symbol
from sympy.simplify import simplify, trigsimp
from sympy.functions.elementary.miscellaneous import sqrt, Max
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.elliptic_integrals import elliptic_e
from .entity import GeometryEntity, GeometrySet
from .exceptions import GeometryError
from .line import Line, Segment, Ray2D, Segment2D, Line2D, LinearEntity3D
from .point import Point, Point2D, Point3D
from .util import idiff, find
from sympy.polys import DomainError, Poly, PolynomialError
from sympy.polys.polyutils import _not_a_coeff, _nsort
from sympy.solvers import solve
from sympy.solvers.solveset import linear_coeffs
from sympy.utilities.misc import filldedent, func_name
from mpmath.libmp.libmpf import prec_to_dps
import random
from .polygon import Polygon, Triangle
def director_circle(self):
    """
        Returns a Circle consisting of all points where two perpendicular
        tangent lines to the ellipse cross each other.

        Returns
        =======

        Circle
            A director circle returned as a geometric object.

        Examples
        ========

        >>> from sympy import Ellipse, Point, symbols
        >>> c = Point(3,8)
        >>> Ellipse(c, 7, 9).director_circle()
        Circle(Point2D(3, 8), sqrt(130))
        >>> a, b = symbols('a b')
        >>> Ellipse(c, a, b).director_circle()
        Circle(Point2D(3, 8), sqrt(a**2 + b**2))

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Director_circle

        """
    return Circle(self.center, sqrt(self.hradius ** 2 + self.vradius ** 2))