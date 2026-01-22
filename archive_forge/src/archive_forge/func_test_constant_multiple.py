from sympy.core.function import Function
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (cosh, sinh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, cos, sin)
from sympy.integrals.integrals import Integral
from sympy.solvers.ode.ode import constantsimp, constant_renumber
from sympy.testing.pytest import XFAIL
def test_constant_multiple():
    assert constant_renumber(constantsimp(C1 * 2 + 2, [C1])) == C1
    assert constant_renumber(constantsimp(x * 2 / C1, [C1])) == C1 * x
    assert constant_renumber(constantsimp(C1 ** 2 * 2 + 2, [C1])) == C1
    assert constant_renumber(constantsimp(sin(2 * C1) + x + sqrt(2), [C1])) == C1 + x
    assert constant_renumber(constantsimp(2 * C1 + C2, [C1, C2])) == C1