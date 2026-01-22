from sympy.polys.polytools import Poly
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable
from sympy.utilities.decorator import public
from .basis import round_two, nilradical_mod_p
from .exceptions import StructureError
from .modules import ModuleEndomorphism, find_min_poly
from .utilities import coeff_search, supplement_a_subspace
@public
def prime_valuation(I, P):
    """
    Compute the *P*-adic valuation for an integral ideal *I*.

    Examples
    ========

    >>> from sympy import QQ
    >>> from sympy.polys.numberfields import prime_valuation
    >>> K = QQ.cyclotomic_field(5)
    >>> P = K.primes_above(5)
    >>> ZK = K.maximal_order()
    >>> print(prime_valuation(25*ZK, P[0]))
    8

    Parameters
    ==========

    I : :py:class:`~.Submodule`
        An integral ideal whose valuation is desired.

    P : :py:class:`~.PrimeIdeal`
        The prime at which to compute the valuation.

    Returns
    =======

    int

    See Also
    ========

    .PrimeIdeal.valuation

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithm 4.8.17.)

    """
    p, ZK = (P.p, P.ZK)
    n, W, d = (ZK.n, ZK.matrix, ZK.denom)
    A = W.convert_to(QQ).inv() * I.matrix * d / I.denom
    A = A.convert_to(ZZ)
    D = A.det()
    if D % p != 0:
        return 0
    beta = P.test_factor()
    f = d ** n // W.det()
    need_complete_test = f % p == 0
    v = 0
    while True:
        A = W * A
        for j in range(n):
            c = ZK.parent(A[:, j], denom=d)
            c *= beta
            c = ZK.represent(c).flat()
            for i in range(n):
                A[i, j] = c[i]
        if A[n - 1, n - 1].element % p != 0:
            break
        A = A / p
        if need_complete_test:
            try:
                A = A.convert_to(ZZ)
            except CoercionFailed:
                break
        else:
            A = A.convert_to(ZZ)
        v += 1
    return v