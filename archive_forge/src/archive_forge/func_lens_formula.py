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
def lens_formula(focal_length=None, u=None, v=None):
    """
    This function provides one of the three parameters
    when two of them are supplied.
    This is valid only for paraxial rays.

    Parameters
    ==========

    focal_length : sympifiable
        Focal length of the mirror.
    u : sympifiable
        Distance of object from the optical center on
        the principal axis.
    v : sympifiable
        Distance of the image from the optical center
        on the principal axis.

    Examples
    ========

    >>> from sympy.physics.optics import lens_formula
    >>> from sympy.abc import f, u, v
    >>> lens_formula(focal_length=f, u=u)
    f*u/(f + u)
    >>> lens_formula(focal_length=f, v=v)
    f*v/(f - v)
    >>> lens_formula(u=u, v=v)
    u*v/(u - v)

    """
    if focal_length and u and v:
        raise ValueError('Please provide only two parameters')
    focal_length = sympify(focal_length)
    u = sympify(u)
    v = sympify(v)
    if u is oo:
        _u = Symbol('u')
    if v is oo:
        _v = Symbol('v')
    if focal_length is oo:
        _f = Symbol('f')
    if focal_length is None:
        if u is oo and v is oo:
            return Limit(Limit(_v * _u / (_u - _v), _u, oo), _v, oo).doit()
        if u is oo:
            return Limit(v * _u / (_u - v), _u, oo).doit()
        if v is oo:
            return Limit(_v * u / (u - _v), _v, oo).doit()
        return v * u / (u - v)
    if u is None:
        if v is oo and focal_length is oo:
            return Limit(Limit(_v * _f / (_f - _v), _v, oo), _f, oo).doit()
        if v is oo:
            return Limit(_v * focal_length / (focal_length - _v), _v, oo).doit()
        if focal_length is oo:
            return Limit(v * _f / (_f - v), _f, oo).doit()
        return v * focal_length / (focal_length - v)
    if v is None:
        if u is oo and focal_length is oo:
            return Limit(Limit(_u * _f / (_u + _f), _u, oo), _f, oo).doit()
        if u is oo:
            return Limit(_u * focal_length / (_u + focal_length), _u, oo).doit()
        if focal_length is oo:
            return Limit(u * _f / (u + _f), _f, oo).doit()
        return u * focal_length / (u + focal_length)