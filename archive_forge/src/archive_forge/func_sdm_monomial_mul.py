from itertools import permutations
from sympy.polys.monomials import (
from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
def sdm_monomial_mul(M, X):
    """
    Multiply tuple ``X`` representing a monomial of `K[X]` into the tuple
    ``M`` representing a monomial of `F`.

    Examples
    ========

    Multiplying `xy^3` into `x f_1` yields `x^2 y^3 f_1`:

    >>> from sympy.polys.distributedmodules import sdm_monomial_mul
    >>> sdm_monomial_mul((1, 1, 0), (1, 3))
    (1, 2, 3)
    """
    return (M[0],) + monomial_mul(X, M[1:])