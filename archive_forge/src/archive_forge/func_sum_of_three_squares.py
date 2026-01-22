from sympy.core.add import Add
from sympy.core.assumptions import check_assumptions
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.numbers import igcdex, ilcm, igcd
from sympy.core.power import integer_nthroot, isqrt
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import _sympify
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import MutableDenseMatrix as Matrix
from sympy.ntheory.factor_ import (
from sympy.ntheory.generate import nextprime
from sympy.ntheory.primetest import is_square, isprime
from sympy.ntheory.residue_ntheory import sqrt_mod
from sympy.polys.polyerrors import GeneratorsNeeded
from sympy.polys.polytools import Poly, factor_list
from sympy.simplify.simplify import signsimp
from sympy.solvers.solveset import solveset_real
from sympy.utilities import numbered_symbols
from sympy.utilities.misc import as_int, filldedent
from sympy.utilities.iterables import (is_sequence, subsets, permute_signs,
def sum_of_three_squares(n):
    """
    Returns a 3-tuple $(a, b, c)$ such that $a^2 + b^2 + c^2 = n$ and
    $a, b, c \\geq 0$.

    Returns None if $n = 4^a(8m + 7)$ for some `a, m \\in \\mathbb{Z}`. See
    [1]_ for more details.

    Usage
    =====

    ``sum_of_three_squares(n)``: Here ``n`` is a non-negative integer.

    Examples
    ========

    >>> from sympy.solvers.diophantine.diophantine import sum_of_three_squares
    >>> sum_of_three_squares(44542)
    (18, 37, 207)

    References
    ==========

    .. [1] Representing a number as a sum of three squares, [online],
        Available: https://schorn.ch/lagrange.html

    See Also
    ========

    sum_of_squares()
    """
    special = {1: (1, 0, 0), 2: (1, 1, 0), 3: (1, 1, 1), 10: (1, 3, 0), 34: (3, 3, 4), 58: (3, 7, 0), 85: (6, 7, 0), 130: (3, 11, 0), 214: (3, 6, 13), 226: (8, 9, 9), 370: (8, 9, 15), 526: (6, 7, 21), 706: (15, 15, 16), 730: (1, 27, 0), 1414: (6, 17, 33), 1906: (13, 21, 36), 2986: (21, 32, 39), 9634: (56, 57, 57)}
    v = 0
    if n == 0:
        return (0, 0, 0)
    v = multiplicity(4, n)
    n //= 4 ** v
    if n % 8 == 7:
        return
    if n in special.keys():
        x, y, z = special[n]
        return _sorted_tuple(2 ** v * x, 2 ** v * y, 2 ** v * z)
    s, _exact = integer_nthroot(n, 2)
    if _exact:
        return (2 ** v * s, 0, 0)
    x = None
    if n % 8 == 3:
        s = s if _odd(s) else s - 1
        for x in range(s, -1, -2):
            N = (n - x ** 2) // 2
            if isprime(N):
                y, z = prime_as_sum_of_two_squares(N)
                return _sorted_tuple(2 ** v * x, 2 ** v * (y + z), 2 ** v * abs(y - z))
        return
    if n % 8 in (2, 6):
        s = s if _odd(s) else s - 1
    else:
        s = s - 1 if _odd(s) else s
    for x in range(s, -1, -2):
        N = n - x ** 2
        if isprime(N):
            y, z = prime_as_sum_of_two_squares(N)
            return _sorted_tuple(2 ** v * x, 2 ** v * y, 2 ** v * z)