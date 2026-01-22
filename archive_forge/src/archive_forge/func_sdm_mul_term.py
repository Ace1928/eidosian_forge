from itertools import permutations
from sympy.polys.monomials import (
from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
def sdm_mul_term(f, term, O, K):
    """
    Multiply a distributed module element ``f`` by a (polynomial) term ``term``.

    Multiplication of coefficients is done over the ground field ``K``, and
    monomials are ordered according to ``O``.

    Examples
    ========

    `0 f_1 = 0`

    >>> from sympy.polys.distributedmodules import sdm_mul_term
    >>> from sympy.polys import lex, QQ
    >>> sdm_mul_term([((1, 0, 0), QQ(1))], ((0, 0), QQ(0)), lex, QQ)
    []

    `x 0 = 0`

    >>> sdm_mul_term([], ((1, 0), QQ(1)), lex, QQ)
    []

    `(x) (f_1) = xf_1`

    >>> sdm_mul_term([((1, 0, 0), QQ(1))], ((1, 0), QQ(1)), lex, QQ)
    [((1, 1, 0), 1)]

    `(2xy) (3x f_1 + 4y f_2) = 8xy^2 f_2 + 6x^2y f_1`

    >>> f = [((2, 0, 1), QQ(4)), ((1, 1, 0), QQ(3))]
    >>> sdm_mul_term(f, ((1, 1), QQ(2)), lex, QQ)
    [((2, 1, 2), 8), ((1, 2, 1), 6)]
    """
    X, c = term
    if not f or not c:
        return []
    elif K.is_one(c):
        return [(sdm_monomial_mul(f_M, X), f_c) for f_M, f_c in f]
    else:
        return [(sdm_monomial_mul(f_M, X), f_c * c) for f_M, f_c in f]