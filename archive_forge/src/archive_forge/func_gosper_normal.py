from sympy.core import S, Dummy, symbols
from sympy.polys import Poly, parallel_poly_from_expr, factor
from sympy.utilities.iterables import is_sequence
def gosper_normal(f, g, n, polys=True):
    """
    Compute the Gosper's normal form of ``f`` and ``g``.

    Explanation
    ===========

    Given relatively prime univariate polynomials ``f`` and ``g``,
    rewrite their quotient to a normal form defined as follows:

    .. math::
        \\frac{f(n)}{g(n)} = Z \\cdot \\frac{A(n) C(n+1)}{B(n) C(n)}

    where ``Z`` is an arbitrary constant and ``A``, ``B``, ``C`` are
    monic polynomials in ``n`` with the following properties:

    1. `\\gcd(A(n), B(n+h)) = 1 \\forall h \\in \\mathbb{N}`
    2. `\\gcd(B(n), C(n+1)) = 1`
    3. `\\gcd(A(n), C(n)) = 1`

    This normal form, or rational factorization in other words, is a
    crucial step in Gosper's algorithm and in solving of difference
    equations. It can be also used to decide if two hypergeometric
    terms are similar or not.

    This procedure will return a tuple containing elements of this
    factorization in the form ``(Z*A, B, C)``.

    Examples
    ========

    >>> from sympy.concrete.gosper import gosper_normal
    >>> from sympy.abc import n

    >>> gosper_normal(4*n+5, 2*(4*n+1)*(2*n+3), n, polys=False)
    (1/4, n + 3/2, n + 1/4)

    """
    (p, q), opt = parallel_poly_from_expr((f, g), n, field=True, extension=True)
    a, A = (p.LC(), p.monic())
    b, B = (q.LC(), q.monic())
    C, Z = (A.one, a / b)
    h = Dummy('h')
    D = Poly(n + h, n, h, domain=opt.domain)
    R = A.resultant(B.compose(D))
    roots = set(R.ground_roots().keys())
    for r in set(roots):
        if not r.is_Integer or r < 0:
            roots.remove(r)
    for i in sorted(roots):
        d = A.gcd(B.shift(+i))
        A = A.quo(d)
        B = B.quo(d.shift(-i))
        for j in range(1, i + 1):
            C *= d.shift(-j)
    A = A.mul_ground(Z)
    if not polys:
        A = A.as_expr()
        B = B.as_expr()
        C = C.as_expr()
    return (A, B, C)