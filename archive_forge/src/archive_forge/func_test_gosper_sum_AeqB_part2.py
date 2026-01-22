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
def test_gosper_sum_AeqB_part2():
    f2a = n ** 2 * a ** n
    f2b = (n - r / 2) * binomial(r, n)
    f2c = factorial(n - 1) ** 2 / (factorial(n - x) * factorial(n + x))
    g2a = -a * (a + 1) / (a - 1) ** 3 + a ** (m + 1) * (a ** 2 * m ** 2 - 2 * a * m ** 2 + m ** 2 - 2 * a * m + 2 * m + a + 1) / (a - 1) ** 3
    g2b = (m - r) * binomial(r, m) / 2
    ff = factorial(1 - x) * factorial(1 + x)
    g2c = 1 / ff * (1 - 1 / x ** 2) + factorial(m) ** 2 / (x ** 2 * factorial(m - x) * factorial(m + x))
    g = gosper_sum(f2a, (n, 0, m))
    assert g is not None and simplify(g - g2a) == 0
    g = gosper_sum(f2b, (n, 0, m))
    assert g is not None and simplify(g - g2b) == 0
    g = gosper_sum(f2c, (n, 1, m))
    assert g is not None and simplify(g - g2c) == 0