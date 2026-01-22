from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import (binomial, factorial)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.gamma_functions import gamma
from sympy.polys.polytools import Poly
from sympy.simplify.simplify import simplify
from sympy.concrete.gosper import gosper_normal, gosper_sum, gosper_term
from sympy.abc import a, b, j, k, m, n, r, x
def test_gosper_sum_indefinite():
    assert gosper_sum(k, k) == k * (k - 1) / 2
    assert gosper_sum(k ** 2, k) == k * (k - 1) * (2 * k - 1) / 6
    assert gosper_sum(1 / (k * (k + 1)), k) == -1 / k
    assert gosper_sum(-(27 * k ** 4 + 158 * k ** 3 + 430 * k ** 2 + 678 * k + 445) * gamma(2 * k + 4) / (3 * (3 * k + 7) * gamma(3 * k + 6)), k) == (3 * k + 5) * (k ** 2 + 2 * k + 5) * gamma(2 * k + 4) / gamma(3 * k + 6)