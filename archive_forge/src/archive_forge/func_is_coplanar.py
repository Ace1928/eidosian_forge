from sympy.core import Dummy, Rational, S, Symbol
from sympy.core.symbol import _symbol
from sympy.functions.elementary.trigonometric import cos, sin, acos, asin, sqrt
from .entity import GeometryEntity
from .line import (Line, Ray, Segment, Line3D, LinearEntity, LinearEntity3D,
from .point import Point, Point3D
from sympy.matrices import Matrix
from sympy.polys.polytools import cancel
from sympy.solvers import solve, linsolve
from sympy.utilities.iterables import uniq, is_sequence
from sympy.utilities.misc import filldedent, func_name, Undecidable
from mpmath.libmp.libmpf import prec_to_dps
import random
def is_coplanar(self, o):
    """ Returns True if `o` is coplanar with self, else False.

        Examples
        ========

        >>> from sympy import Plane
        >>> o = (0, 0, 0)
        >>> p = Plane(o, (1, 1, 1))
        >>> p2 = Plane(o, (2, 2, 2))
        >>> p == p2
        False
        >>> p.is_coplanar(p2)
        True
        """
    if isinstance(o, Plane):
        return not cancel(self.equation(x, y, z) / o.equation(x, y, z)).has(x, y, z)
    if isinstance(o, Point3D):
        return o in self
    elif isinstance(o, LinearEntity3D):
        return all((i in self for i in self))
    elif isinstance(o, GeometryEntity):
        return all((i == 0 for i in self.normal_vector[:2]))