from sympy.core.numbers import oo
from sympy.core import igcd
from sympy.polys.monomials import monomial_min, monomial_div
from sympy.polys.orderings import monomial_key
import random
def dmp_list_terms(f, u, K, order=None):
    """
    List all non-zero terms from ``f`` in the given order ``order``.

    Examples
    ========

    >>> from sympy.polys.domains import ZZ
    >>> from sympy.polys.densebasic import dmp_list_terms

    >>> f = ZZ.map([[1, 1], [2, 3]])

    >>> dmp_list_terms(f, 1, ZZ)
    [((1, 1), 1), ((1, 0), 1), ((0, 1), 2), ((0, 0), 3)]
    >>> dmp_list_terms(f, 1, ZZ, order='grevlex')
    [((1, 1), 1), ((1, 0), 1), ((0, 1), 2), ((0, 0), 3)]

    """

    def sort(terms, O):
        return sorted(terms, key=lambda term: O(term[0]), reverse=True)
    terms = _rec_list_terms(f, u, ())
    if not terms:
        return [((0,) * (u + 1), K.zero)]
    if order is None:
        return terms
    else:
        return sort(terms, monomial_key(order))