from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def dup_cauchy_upper_bound(f, K):
    """
    Compute the Cauchy upper bound on the absolute value of all roots of f,
    real or complex.

    References
    ==========
    .. [1] https://en.wikipedia.org/wiki/Geometrical_properties_of_polynomial_roots#Lagrange's_and_Cauchy's_bounds
    """
    n = dup_degree(f)
    if n < 1:
        raise PolynomialError('Polynomial has no roots.')
    if K.is_ZZ:
        L = K.get_field()
        f, K = (dup_convert(f, K, L), L)
    elif not K.is_QQ or K.is_RR or K.is_CC:
        raise DomainError('Cauchy bound not supported over %s' % K)
    else:
        f = f[:]
    while K.is_zero(f[-1]):
        f.pop()
    if len(f) == 1:
        return K.zero
    lc = f[0]
    return K.one + max((abs(n / lc) for n in f[1:]))