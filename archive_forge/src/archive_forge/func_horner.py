from sympy.core import S, Basic, symbols, Dummy
from sympy.polys.polyerrors import (
from sympy.polys.polyoptions import allowed_flags, build_options
from sympy.polys.polytools import poly_from_expr, Poly
from sympy.polys.specialpolys import (
from sympy.polys.rings import sring
from sympy.utilities import numbered_symbols, take, public
@public
def horner(f, *gens, **args):
    """
    Rewrite a polynomial in Horner form.

    Among other applications, evaluation of a polynomial at a point is optimal
    when it is applied using the Horner scheme ([1]).

    Examples
    ========

    >>> from sympy.polys.polyfuncs import horner
    >>> from sympy.abc import x, y, a, b, c, d, e

    >>> horner(9*x**4 + 8*x**3 + 7*x**2 + 6*x + 5)
    x*(x*(x*(9*x + 8) + 7) + 6) + 5

    >>> horner(a*x**4 + b*x**3 + c*x**2 + d*x + e)
    e + x*(d + x*(c + x*(a*x + b)))

    >>> f = 4*x**2*y**2 + 2*x**2*y + 2*x*y**2 + x*y

    >>> horner(f, wrt=x)
    x*(x*y*(4*y + 2) + y*(2*y + 1))

    >>> horner(f, wrt=y)
    y*(x*y*(4*x + 2) + x*(2*x + 1))

    References
    ==========
    [1] - https://en.wikipedia.org/wiki/Horner_scheme

    """
    allowed_flags(args, [])
    try:
        F, opt = poly_from_expr(f, *gens, **args)
    except PolificationFailed as exc:
        return exc.expr
    form, gen = (S.Zero, F.gen)
    if F.is_univariate:
        for coeff in F.all_coeffs():
            form = form * gen + coeff
    else:
        F, gens = (Poly(F, gen), gens[1:])
        for coeff in F.all_coeffs():
            form = form * gen + horner(coeff, *gens, **args)
    return form