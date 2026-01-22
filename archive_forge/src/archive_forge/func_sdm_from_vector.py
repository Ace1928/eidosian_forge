from itertools import permutations
from sympy.polys.monomials import (
from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
def sdm_from_vector(vec, O, K, **opts):
    """
    Create an sdm from an iterable of expressions.

    Coefficients are created in the ground field ``K``, and terms are ordered
    according to monomial order ``O``. Named arguments are passed on to the
    polys conversion code and can be used to specify for example generators.

    Examples
    ========

    >>> from sympy.polys.distributedmodules import sdm_from_vector
    >>> from sympy.abc import x, y, z
    >>> from sympy.polys import QQ, lex
    >>> sdm_from_vector([x**2+y**2, 2*z], lex, QQ)
    [((1, 0, 0, 1), 2), ((0, 2, 0, 0), 1), ((0, 0, 2, 0), 1)]
    """
    dics, gens = parallel_dict_from_expr(sympify(vec), **opts)
    dic = {}
    for i, d in enumerate(dics):
        for k, v in d.items():
            dic[(i,) + k] = K.convert(v)
    return sdm_from_dict(dic, O)