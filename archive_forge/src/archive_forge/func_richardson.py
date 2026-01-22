from sympy.core.numbers import Integer
from sympy.core.singleton import S
from sympy.functions.combinatorial.factorials import factorial
def richardson(A, k, n, N):
    """
    Calculate an approximation for lim k->oo A(k) using Richardson
    extrapolation with the terms A(n), A(n+1), ..., A(n+N+1).
    Choosing N ~= 2*n often gives good results.

    Examples
    ========

    A simple example is to calculate exp(1) using the limit definition.
    This limit converges slowly; n = 100 only produces two accurate
    digits:

        >>> from sympy.abc import n
        >>> e = (1 + 1/n)**n
        >>> print(round(e.subs(n, 100).evalf(), 10))
        2.7048138294

    Richardson extrapolation with 11 appropriately chosen terms gives
    a value that is accurate to the indicated precision:

        >>> from sympy import E
        >>> from sympy.series.acceleration import richardson
        >>> print(round(richardson(e, n, 10, 20).evalf(), 10))
        2.7182818285
        >>> print(round(E.evalf(), 10))
        2.7182818285

    Another useful application is to speed up convergence of series.
    Computing 100 terms of the zeta(2) series 1/k**2 yields only
    two accurate digits:

        >>> from sympy.abc import k, n
        >>> from sympy import Sum
        >>> A = Sum(k**-2, (k, 1, n))
        >>> print(round(A.subs(n, 100).evalf(), 10))
        1.6349839002

    Richardson extrapolation performs much better:

        >>> from sympy import pi
        >>> print(round(richardson(A, n, 10, 20).evalf(), 10))
        1.6449340668
        >>> print(round(((pi**2)/6).evalf(), 10))     # Exact value
        1.6449340668

    """
    s = S.Zero
    for j in range(0, N + 1):
        s += A.subs(k, Integer(n + j)).doit() * (n + j) ** N * S.NegativeOne ** (j + N) / (factorial(j) * factorial(N - j))
    return s