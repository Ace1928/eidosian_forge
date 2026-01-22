from sympy.core.numbers import (Float, I, oo, pi, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan2, cos, sin, tan)
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import cancel
from sympy.series.limits import Limit
from sympy.geometry.line import Ray3D
from sympy.geometry.util import intersection
from sympy.geometry.plane import Plane
from sympy.utilities.iterables import is_sequence
from .medium import Medium
def brewster_angle(medium1, medium2):
    """
    This function calculates the Brewster's angle of incidence to Medium 2 from
    Medium 1 in radians.

    Parameters
    ==========

    medium 1 : Medium or sympifiable
        Refractive index of Medium 1
    medium 2 : Medium or sympifiable
        Refractive index of Medium 1

    Examples
    ========

    >>> from sympy.physics.optics import brewster_angle
    >>> brewster_angle(1, 1.33)
    0.926093295503462

    """
    n1 = refractive_index_of_medium(medium1)
    n2 = refractive_index_of_medium(medium2)
    return atan2(n2, n1)