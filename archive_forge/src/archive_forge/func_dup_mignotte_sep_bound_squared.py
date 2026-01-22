from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def dup_mignotte_sep_bound_squared(f, K):
    """
    Return the square of the Mignotte lower bound on separation between
    distinct roots of f. The square is returned so that the bound lies in
    K or its quotient field.

    References
    ==========

    .. [1] Mignotte, Maurice. "Some useful bounds." Computer algebra.
        Springer, Vienna, 1982. 259-263.
        https://people.dm.unipi.it/gianni/AC-EAG/Mignotte.pdf
    """
    n = dup_degree(f)
    if n < 2:
        raise PolynomialError('Polynomials of degree < 2 have no distinct roots.')
    if K.is_ZZ:
        L = K.get_field()
        f, K = (dup_convert(f, K, L), L)
    elif not K.is_QQ or K.is_RR or K.is_CC:
        raise DomainError('Mignotte bound not supported over %s' % K)
    D = dup_discriminant(f, K)
    l2sq = dup_l2_norm_squared(f, K)
    return K(3) * K.abs(D) / (K(n) ** (n + 1) * l2sq ** (n - 1))