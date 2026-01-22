from ..pari import pari
import string
from itertools import combinations, combinations_with_replacement, product
def character_variety_ideal(gens, rels=None):
    """
    sage: M = Manifold('m004')
    sage: I = character_variety_ideal(M.fundamental_group())
    sage: I.dimension()
    1
    sage: len(I.radical().primary_decomposition())
    2
    """
    presentation = character_variety(gens, rels)
    from sage.all import PolynomialRing, QQ
    R = PolynomialRing(QQ, [repr(v) for v in presentation.gens])
    return R.ideal([R(p) for p in presentation.rels])