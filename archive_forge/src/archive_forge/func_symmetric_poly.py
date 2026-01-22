from sympy.core import Add, Mul, Symbol, sympify, Dummy, symbols
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.ntheory import nextprime
from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.domains import ZZ
from sympy.polys.factortools import dup_zz_cyclotomic_poly
from sympy.polys.polyclasses import DMP
from sympy.polys.polytools import Poly, PurePoly
from sympy.polys.polyutils import _analyze_gens
from sympy.utilities import subsets, public, filldedent
from sympy.polys.rings import ring
@public
def symmetric_poly(n, *gens, polys=False):
    """
    Generates symmetric polynomial of order `n`.

    Parameters
    ==========

    polys: bool, optional (default: False)
        Returns a Poly object when ``polys=True``, otherwise
        (default) returns an expression.
    """
    gens = _analyze_gens(gens)
    if n < 0 or n > len(gens) or (not gens):
        raise ValueError('Cannot generate symmetric polynomial of order %s for %s' % (n, gens))
    elif not n:
        poly = S.One
    else:
        poly = Add(*[Mul(*s) for s in subsets(gens, int(n))])
    return Poly(poly, *gens) if polys else poly