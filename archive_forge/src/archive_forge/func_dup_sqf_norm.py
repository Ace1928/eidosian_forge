from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
def dup_sqf_norm(f, K):
    """
    Square-free norm of ``f`` in ``K[x]``, useful over algebraic domains.

    Returns ``s``, ``f``, ``r``, such that ``g(x) = f(x-sa)`` and ``r(x) = Norm(g(x))``
    is a square-free polynomial over K, where ``a`` is the algebraic extension of ``K``.

    Examples
    ========

    >>> from sympy.polys import ring, QQ
    >>> from sympy import sqrt

    >>> K = QQ.algebraic_field(sqrt(3))
    >>> R, x = ring("x", K)
    >>> _, X = ring("x", QQ)

    >>> s, f, r = R.dup_sqf_norm(x**2 - 2)

    >>> s == 1
    True
    >>> f == x**2 + K([QQ(-2), QQ(0)])*x + 1
    True
    >>> r == X**4 - 10*X**2 + 1
    True

    """
    if not K.is_Algebraic:
        raise DomainError('ground domain must be algebraic')
    s, g = (0, dmp_raise(K.mod.rep, 1, 0, K.dom))
    while True:
        h, _ = dmp_inject(f, 0, K, front=True)
        r = dmp_resultant(g, h, 1, K.dom)
        if dup_sqf_p(r, K.dom):
            break
        else:
            f, s = (dup_shift(f, -K.unit, K), s + 1)
    return (s, f, r)