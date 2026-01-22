from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (E, I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import sin
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.simplify.powsimp import (powdenest, powsimp)
from sympy.simplify.simplify import (signsimp, simplify)
from sympy.core.symbol import Str
from sympy.abc import x, y, z, a, b
def test_issue_19627():
    assert powdenest(sqrt(sin(x) ** 2), force=True) == sin(x)
    assert powdenest((x ** (S.Half / y)) ** (2 * y), force=True) == x
    from sympy.core.function import expand_power_base
    e = 1 - a
    expr = (exp(z / e) * x ** (b / e) * y ** ((1 - b) / e)) ** e
    assert powdenest(expand_power_base(expr, force=True), force=True) == x ** b * y ** (1 - b) * exp(z)