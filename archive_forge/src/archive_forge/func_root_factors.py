import math
from functools import reduce
from sympy.core import S, I, pi
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.logic import fuzzy_not
from sympy.core.mul import expand_2arg, Mul
from sympy.core.numbers import Rational, igcd, comp
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol, symbols
from sympy.core.sympify import sympify
from sympy.functions import exp, im, cos, acos, Piecewise
from sympy.functions.elementary.miscellaneous import root, sqrt
from sympy.ntheory import divisors, isprime, nextprime
from sympy.polys.domains import EX
from sympy.polys.polyerrors import (PolynomialError, GeneratorsNeeded,
from sympy.polys.polyquinticconst import PolyQuintic
from sympy.polys.polytools import Poly, cancel, factor, gcd_list, discriminant
from sympy.polys.rationaltools import together
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.utilities import public
from sympy.utilities.misc import filldedent
def root_factors(f, *gens, filter=None, **args):
    """
    Returns all factors of a univariate polynomial.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.polys.polyroots import root_factors

    >>> root_factors(x**2 - y, x)
    [x - sqrt(y), x + sqrt(y)]

    """
    args = dict(args)
    F = Poly(f, *gens, **args)
    if not F.is_Poly:
        return [f]
    if F.is_multivariate:
        raise ValueError('multivariate polynomials are not supported')
    x = F.gens[0]
    zeros = roots(F, filter=filter)
    if not zeros:
        factors = [F]
    else:
        factors, N = ([], 0)
        for r, n in ordered(zeros.items()):
            factors, N = (factors + [Poly(x - r, x)] * n, N + n)
        if N < F.degree():
            G = reduce(lambda p, q: p * q, factors)
            factors.append(F.quo(G))
    if not isinstance(f, Poly):
        factors = [f.as_expr() for f in factors]
    return factors