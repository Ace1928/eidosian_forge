from itertools import permutations
from sympy.polys.monomials import (
from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
def sdm_monomial_deg(M):
    """
    Return the total degree of ``M``.

    Examples
    ========

    For example, the total degree of `x^2 y f_5` is 3:

    >>> from sympy.polys.distributedmodules import sdm_monomial_deg
    >>> sdm_monomial_deg((5, 2, 1))
    3
    """
    return monomial_deg(M[1:])