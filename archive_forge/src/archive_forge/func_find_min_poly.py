from sympy.core.numbers import igcd, ilcm
from sympy.core.symbol import Dummy
from sympy.polys.polyclasses import ANP
from sympy.polys.polytools import Poly
from sympy.polys.densetools import dup_clear_denoms
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.finitefield import FF
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.domainmatrix import DomainMatrix
from sympy.polys.matrices.exceptions import DMBadInputError
from sympy.polys.matrices.normalforms import hermite_normal_form
from sympy.polys.polyerrors import CoercionFailed, UnificationFailed
from sympy.polys.polyutils import IntegerPowerable
from .exceptions import ClosureFailure, MissingUnityError, StructureError
from .utilities import AlgIntPowers, is_rat, get_num_denom
def find_min_poly(alpha, domain, x=None, powers=None):
    """
    Find a polynomial of least degree (not necessarily irreducible) satisfied
    by an element of a finitely-generated ring with unity.

    Examples
    ========

    For the $n$th cyclotomic field, $n$ an odd prime, consider the quadratic
    equation whose roots are the two periods of length $(n-1)/2$. Article 356
    of Gauss tells us that we should get $x^2 + x - (n-1)/4$ or
    $x^2 + x + (n+1)/4$ according to whether $n$ is 1 or 3 mod 4, respectively.

    >>> from sympy import Poly, cyclotomic_poly, primitive_root, QQ
    >>> from sympy.abc import x
    >>> from sympy.polys.numberfields.modules import PowerBasis, find_min_poly
    >>> n = 13
    >>> g = primitive_root(n)
    >>> C = PowerBasis(Poly(cyclotomic_poly(n, x)))
    >>> ee = [g**(2*k+1) % n for k in range((n-1)//2)]
    >>> eta = sum(C(e) for e in ee)
    >>> print(find_min_poly(eta, QQ, x=x).as_expr())
    x**2 + x - 3
    >>> n = 19
    >>> g = primitive_root(n)
    >>> C = PowerBasis(Poly(cyclotomic_poly(n, x)))
    >>> ee = [g**(2*k+2) % n for k in range((n-1)//2)]
    >>> eta = sum(C(e) for e in ee)
    >>> print(find_min_poly(eta, QQ, x=x).as_expr())
    x**2 + x + 5

    Parameters
    ==========

    alpha : :py:class:`~.ModuleElement`
        The element whose min poly is to be found, and whose module has
        multiplication and starts with unity.

    domain : :py:class:`~.Domain`
        The desired domain of the polynomial.

    x : :py:class:`~.Symbol`, optional
        The desired variable for the polynomial.

    powers : list, optional
        If desired, pass an empty list. The powers of *alpha* (as
        :py:class:`~.ModuleElement` instances) from the zeroth up to the degree
        of the min poly will be recorded here, as we compute them.

    Returns
    =======

    :py:class:`~.Poly`, ``None``
        The minimal polynomial for alpha, or ``None`` if no polynomial could be
        found over the desired domain.

    Raises
    ======

    MissingUnityError
        If the module to which alpha belongs does not start with unity.
    ClosureFailure
        If the module to which alpha belongs is not closed under
        multiplication.

    """
    R = alpha.module
    if not R.starts_with_unity():
        raise MissingUnityError('alpha must belong to finitely generated ring with unity.')
    if powers is None:
        powers = []
    one = R(0)
    powers.append(one)
    powers_matrix = one.column(domain=domain)
    ak = alpha
    m = None
    for k in range(1, R.n + 1):
        powers.append(ak)
        ak_col = ak.column(domain=domain)
        try:
            X = powers_matrix._solve(ak_col)[0]
        except DMBadInputError:
            powers_matrix = powers_matrix.hstack(ak_col)
            ak *= alpha
        else:
            coeffs = [1] + [-c for c in reversed(X.to_list_flat())]
            x = x or Dummy('x')
            if domain.is_FF:
                m = Poly(coeffs, x, modulus=domain.mod)
            else:
                m = Poly(coeffs, x, domain=domain)
            break
    return m