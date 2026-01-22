from itertools import permutations
from sympy.polys.monomials import (
from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
def sdm_strip(f):
    """Remove terms with zero coefficients from ``f`` in ``K[X]``. """
    return [(monom, coeff) for monom, coeff in f if coeff]