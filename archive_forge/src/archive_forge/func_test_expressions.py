from sympy.core.function import Function
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cot, sin, tan)
from sympy.series.residues import residue
from sympy.testing.pytest import XFAIL, raises
from sympy.abc import x, z, a, s, k
def test_expressions():
    assert residue(1 / (x + 1), x, 0) == 0
    assert residue(1 / (x + 1), x, -1) == 1
    assert residue(1 / (x ** 2 + 1), x, -1) == 0
    assert residue(1 / (x ** 2 + 1), x, I) == -I / 2
    assert residue(1 / (x ** 2 + 1), x, -I) == I / 2
    assert residue(1 / (x ** 4 + 1), x, 0) == 0
    assert residue(1 / (x ** 4 + 1), x, exp(I * pi / 4)).equals(-(Rational(1, 4) + I / 4) / sqrt(2))
    assert residue(1 / (x ** 2 + a ** 2) ** 2, x, a * I) == -I / 4 / a ** 3