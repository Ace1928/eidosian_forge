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
def test_accum_bounds():
    assert limit_seq((-1) ** n, n) == AccumulationBounds(-1, 1)
    assert limit_seq(cos(pi * n), n) == AccumulationBounds(-1, 1)
    assert limit_seq(sin(pi * n / 2) ** 2, n) == AccumulationBounds(0, 1)
    assert limit_seq(2 * (-3) ** n / (n + 3 ** n), n) == AccumulationBounds(-2, 2)
    assert limit_seq(3 * n / (n + 1) + 2 * (-1) ** n, n) == AccumulationBounds(1, 5)