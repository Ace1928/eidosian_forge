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
def test_issue_5728():
    b = x * sqrt(y)
    a = sqrt(b)
    c = sqrt(sqrt(x) * y)
    assert powsimp(a * b) == sqrt(b) ** 3
    assert powsimp(a * b ** 2 * sqrt(y)) == sqrt(y) * a ** 5
    assert powsimp(a * x ** 2 * c ** 3 * y) == c ** 3 * a ** 5
    assert powsimp(a * x * c ** 3 * y ** 2) == c ** 7 * a
    assert powsimp(x * c ** 3 * y ** 2) == c ** 7
    assert powsimp(x * c ** 3 * y) == x * y * c ** 3
    assert powsimp(sqrt(x) * c ** 3 * y) == c ** 5
    assert powsimp(sqrt(x) * a ** 3 * sqrt(y)) == sqrt(x) * sqrt(y) * a ** 3
    assert powsimp(Mul(sqrt(x) * c ** 3 * sqrt(y), y, evaluate=False)) == sqrt(x) * sqrt(y) ** 3 * c ** 3
    assert powsimp(a ** 2 * a * x ** 2 * y) == a ** 7
    b = x ** y * y
    a = b * sqrt(b)
    assert a.is_Mul is True
    assert powsimp(a) == sqrt(b) ** 3
    a = x * exp(y * Rational(2, 3))
    assert powsimp(a * sqrt(a)) == sqrt(a) ** 3
    assert powsimp(a ** 2 * sqrt(a)) == sqrt(a) ** 5
    assert powsimp(a ** 2 * sqrt(sqrt(a))) == sqrt(sqrt(a)) ** 9