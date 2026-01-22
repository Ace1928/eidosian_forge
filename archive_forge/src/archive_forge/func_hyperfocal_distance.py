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
def hyperfocal_distance(f, N, c):
    """

    Parameters
    ==========

    f: sympifiable
        Focal length of a given lens.

    N: sympifiable
        F-number of a given lens.

    c: sympifiable
        Circle of Confusion (CoC) of a given image format.

    Example
    =======

    >>> from sympy.physics.optics import hyperfocal_distance
    >>> round(hyperfocal_distance(f = 0.5, N = 8, c = 0.0033), 2)
    9.47
    """
    f = sympify(f)
    N = sympify(N)
    c = sympify(c)
    return 1 / (N * c) * f ** 2