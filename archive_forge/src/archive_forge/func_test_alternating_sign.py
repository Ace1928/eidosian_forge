from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.combinatorial.factorials import (binomial, factorial, subfactorial)
from sympy.functions.combinatorial.numbers import (fibonacci, harmonic)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.series.limitseq import limit_seq
from sympy.series.limitseq import difference_delta as dd
from sympy.testing.pytest import raises, XFAIL
from sympy.calculus.accumulationbounds import AccumulationBounds
def test_alternating_sign():
    assert limit_seq((-1) ** n / n ** 2, n) == 0
    assert limit_seq((-2) ** (n + 1) / (n + 3 ** n), n) == 0
    assert limit_seq((2 * n + (-1) ** n) / (n + 1), n) == 2
    assert limit_seq(sin(pi * n), n) == 0
    assert limit_seq(cos(2 * pi * n), n) == 1
    assert limit_seq((S.NegativeOne / 5) ** n, n) == 0
    assert limit_seq(Rational(-1, 5) ** n, n) == 0
    assert limit_seq((I / 3) ** n, n) == 0
    assert limit_seq(sqrt(n) * (I / 2) ** n, n) == 0
    assert limit_seq(n ** 7 * (I / 3) ** n, n) == 0
    assert limit_seq(n / (n + 1) + (I / 2) ** n, n) == 1