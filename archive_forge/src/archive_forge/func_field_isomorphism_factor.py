from sympy.core.add import Add
from sympy.core.numbers import AlgebraicNumber
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify, _sympify
from sympy.ntheory import sieve
from sympy.polys.densetools import dup_eval
from sympy.polys.domains import QQ
from sympy.polys.numberfields.minpoly import _choose_factor, minimal_polynomial
from sympy.polys.polyerrors import IsomorphismFailed
from sympy.polys.polytools import Poly, PurePoly, factor_list
from sympy.utilities import public
from mpmath import MPContext
def field_isomorphism_factor(a, b):
    """Construct field isomorphism via factorization. """
    _, factors = factor_list(a.minpoly, extension=b)
    for f, _ in factors:
        if f.degree() == 1:
            c = -f.rep.TC()
            coeffs = c.to_sympy_list()
            d, terms = (len(coeffs) - 1, [])
            for i, coeff in enumerate(coeffs):
                terms.append(coeff * b.root ** (d - i))
            r = Add(*terms)
            if a.minpoly.same_root(r, a):
                return coeffs
    return None