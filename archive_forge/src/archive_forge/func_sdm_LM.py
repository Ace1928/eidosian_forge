from itertools import permutations
from sympy.polys.monomials import (
from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
def sdm_LM(f):
    """
    Returns the leading monomial of ``f``.

    Only valid if `f \\ne 0`.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_LM, sdm_from_dict
    >>> from sympy.polys import QQ, lex
    >>> dic = {(1, 2, 3): QQ(1), (4, 0, 0): QQ(1), (4, 0, 1): QQ(1)}
    >>> sdm_LM(sdm_from_dict(dic, lex))
    (4, 0, 1)
    """
    return f[0][0]