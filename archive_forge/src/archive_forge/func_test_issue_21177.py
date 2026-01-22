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
def test_issue_21177():
    r = -sqrt(3) * tanh(sqrt(3) * pi / 2) / 3
    a = residue(cot(pi * x) / ((x - 1) * (x - 2) + 1), x, S(3) / 2 - sqrt(3) * I / 2)
    b = residue(cot(pi * x) / (x ** 2 - 3 * x + 3), x, S(3) / 2 - sqrt(3) * I / 2)
    assert a == r
    assert (b - a).cancel() == 0