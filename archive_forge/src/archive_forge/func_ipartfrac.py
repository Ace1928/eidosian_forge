from __future__ import annotations
from typing import Callable
from functools import reduce
from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.numbers import igcdex, Integer
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.core.cache import cacheit
def ipartfrac(*denoms: int) -> tuple[int, ...]:
    """Compute the the partial fraction decomposition.

    Explanation
    ===========

    Given a rational number $\\frac{1}{q_1 \\cdots q_n}$ where all
    $q_1, \\cdots, q_n$ are pairwise coprime,

    A partial fraction decomposition is defined as

    .. math::
        \\frac{1}{q_1 \\cdots q_n} = \\frac{p_1}{q_1} + \\cdots + \\frac{p_n}{q_n}

    And it can be derived from solving the following diophantine equation for
    the $p_1, \\cdots, p_n$

    .. math::
        1 = p_1 \\prod_{i \\ne 1}q_i + \\cdots + p_n \\prod_{i \\ne n}q_i

    Where $q_1, \\cdots, q_n$ being pairwise coprime implies
    $\\gcd(\\prod_{i \\ne 1}q_i, \\cdots, \\prod_{i \\ne n}q_i) = 1$,
    which guarantees the existance of the solution.

    It is sufficient to compute partial fraction decomposition only
    for numerator $1$ because partial fraction decomposition for any
    $\\frac{n}{q_1 \\cdots q_n}$ can be easily computed by multiplying
    the result by $n$ afterwards.

    Parameters
    ==========

    denoms : int
        The pairwise coprime integer denominators $q_i$ which defines the
        rational number $\\frac{1}{q_1 \\cdots q_n}$

    Returns
    =======

    tuple[int, ...]
        The list of numerators which semantically corresponds to $p_i$ of the
        partial fraction decomposition
        $\\frac{1}{q_1 \\cdots q_n} = \\frac{p_1}{q_1} + \\cdots + \\frac{p_n}{q_n}$

    Examples
    ========

    >>> from sympy import Rational, Mul
    >>> from sympy.functions.elementary._trigonometric_special import ipartfrac

    >>> denoms = 2, 3, 5
    >>> numers = ipartfrac(2, 3, 5)
    >>> numers
    (1, 7, -14)

    >>> Rational(1, Mul(*denoms))
    1/30
    >>> out = 0
    >>> for n, d in zip(numers, denoms):
    ...    out += Rational(n, d)
    >>> out
    1/30
    """
    if not denoms:
        return ()

    def mul(x: int, y: int) -> int:
        return x * y
    denom = reduce(mul, denoms)
    a = [denom // x for x in denoms]
    h, _ = migcdex(*a)
    return h