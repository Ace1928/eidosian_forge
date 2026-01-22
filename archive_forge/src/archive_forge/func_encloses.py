from __future__ import annotations
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.evalf import EvalfMixin, N
from sympy.core.numbers import oo
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import cos, sin, atan
from sympy.matrices import eye
from sympy.multipledispatch import dispatch
from sympy.printing import sstr
from sympy.sets import Set, Union, FiniteSet
from sympy.sets.handlers.intersection import intersection_sets
from sympy.sets.handlers.union import union_sets
from sympy.solvers.solvers import solve
from sympy.utilities.misc import func_name
from sympy.utilities.iterables import is_sequence
def encloses(self, o):
    """
        Return True if o is inside (not on or outside) the boundaries of self.

        The object will be decomposed into Points and individual Entities need
        only define an encloses_point method for their class.

        See Also
        ========

        sympy.geometry.ellipse.Ellipse.encloses_point
        sympy.geometry.polygon.Polygon.encloses_point

        Examples
        ========

        >>> from sympy import RegularPolygon, Point, Polygon
        >>> t  = Polygon(*RegularPolygon(Point(0, 0), 1, 3).vertices)
        >>> t2 = Polygon(*RegularPolygon(Point(0, 0), 2, 3).vertices)
        >>> t2.encloses(t)
        True
        >>> t.encloses(t2)
        False

        """
    from sympy.geometry.point import Point
    from sympy.geometry.line import Segment, Ray, Line
    from sympy.geometry.ellipse import Ellipse
    from sympy.geometry.polygon import Polygon, RegularPolygon
    if isinstance(o, Point):
        return self.encloses_point(o)
    elif isinstance(o, Segment):
        return all((self.encloses_point(x) for x in o.points))
    elif isinstance(o, (Ray, Line)):
        return False
    elif isinstance(o, Ellipse):
        return self.encloses_point(o.center) and self.encloses_point(Point(o.center.x + o.hradius, o.center.y)) and (not self.intersection(o))
    elif isinstance(o, Polygon):
        if isinstance(o, RegularPolygon):
            if not self.encloses_point(o.center):
                return False
        return all((self.encloses_point(v) for v in o.vertices))
    raise NotImplementedError()