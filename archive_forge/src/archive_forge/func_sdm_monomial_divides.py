from itertools import permutations
from sympy.polys.monomials import (
from sympy.polys.polytools import Poly
from sympy.polys.polyutils import parallel_dict_from_expr
from sympy.core.singleton import S
from sympy.core.sympify import sympify
def sdm_monomial_divides(A, B):
    """
    Does there exist a (polynomial) monomial X such that XA = B?

    Examples
    ========

    Positive examples:

    In the following examples, the monomial is given in terms of x, y and the
    generator(s), f_1, f_2 etc. The tuple form of that monomial is used in
    the call to sdm_monomial_divides.
    Note: the generator appears last in the expression but first in the tuple
    and other factors appear in the same order that they appear in the monomial
    expression.

    `A = f_1` divides `B = f_1`

    >>> from sympy.polys.distributedmodules import sdm_monomial_divides
    >>> sdm_monomial_divides((1, 0, 0), (1, 0, 0))
    True

    `A = f_1` divides `B = x^2 y f_1`

    >>> sdm_monomial_divides((1, 0, 0), (1, 2, 1))
    True

    `A = xy f_5` divides `B = x^2 y f_5`

    >>> sdm_monomial_divides((5, 1, 1), (5, 2, 1))
    True

    Negative examples:

    `A = f_1` does not divide `B = f_2`

    >>> sdm_monomial_divides((1, 0, 0), (2, 0, 0))
    False

    `A = x f_1` does not divide `B = f_1`

    >>> sdm_monomial_divides((1, 1, 0), (1, 0, 0))
    False

    `A = xy^2 f_5` does not divide `B = y f_5`

    >>> sdm_monomial_divides((5, 1, 2), (5, 0, 1))
    False
    """
    return A[0] == B[0] and all((a <= b for a, b in zip(A[1:], B[1:])))