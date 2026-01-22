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
def test_powsimp():
    x, y, z, n = symbols('x,y,z,n')
    f = Function('f')
    assert powsimp(4 ** x * 2 ** (-x) * 2 ** (-x)) == 1
    assert powsimp((-4) ** x * (-2) ** (-x) * 2 ** (-x)) == 1
    assert powsimp(f(4 ** x * 2 ** (-x) * 2 ** (-x))) == f(4 ** x * 2 ** (-x) * 2 ** (-x))
    assert powsimp(f(4 ** x * 2 ** (-x) * 2 ** (-x)), deep=True) == f(1)
    assert exp(x) * exp(y) == exp(x) * exp(y)
    assert powsimp(exp(x) * exp(y)) == exp(x + y)
    assert powsimp(exp(x) * exp(y) * 2 ** x * 2 ** y) == (2 * E) ** (x + y)
    assert powsimp(exp(x) * exp(y) * 2 ** x * 2 ** y, combine='exp') == exp(x + y) * 2 ** (x + y)
    assert powsimp(exp(x) * exp(y) * exp(2) * sin(x) + sin(y) + 2 ** x * 2 ** y) == exp(2 + x + y) * sin(x) + sin(y) + 2 ** (x + y)
    assert powsimp(sin(exp(x) * exp(y))) == sin(exp(x) * exp(y))
    assert powsimp(sin(exp(x) * exp(y)), deep=True) == sin(exp(x + y))
    assert powsimp(x ** 2 * x ** y) == x ** (2 + y)
    assert powsimp((1 + E * exp(E)) * exp(-E), combine='exp', deep=True) == (1 + exp(1 + E)) * exp(-E)
    assert powsimp((1 + E * exp(E)) * exp(-E), deep=True) == (1 + exp(1 + E)) * exp(-E)
    assert powsimp((1 + E * exp(E)) * exp(-E)) == (1 + exp(1 + E)) * exp(-E)
    assert powsimp((1 + E * exp(E)) * exp(-E), combine='exp') == (1 + exp(1 + E)) * exp(-E)
    assert powsimp((1 + E * exp(E)) * exp(-E), combine='base') == (1 + E * exp(E)) * exp(-E)
    x, y = symbols('x,y', nonnegative=True)
    n = Symbol('n', real=True)
    assert powsimp(y ** n * (y / x) ** (-n)) == x ** n
    assert powsimp(x ** (x ** (x * y) * y ** (x * y)) * y ** (x ** (x * y) * y ** (x * y)), deep=True) == (x * y) ** (x * y) ** (x * y)
    assert powsimp(2 ** (2 ** (2 * x) * x), deep=False) == 2 ** (2 ** (2 * x) * x)
    assert powsimp(2 ** (2 ** (2 * x) * x), deep=True) == 2 ** (x * 4 ** x)
    assert powsimp(exp(-x + exp(-x) * exp(-x * log(x))), deep=False, combine='exp') == exp(-x + exp(-x) * exp(-x * log(x)))
    assert powsimp(exp(-x + exp(-x) * exp(-x * log(x))), deep=False, combine='exp') == exp(-x + exp(-x) * exp(-x * log(x)))
    assert powsimp((x + y) / (3 * z), deep=False, combine='exp') == (x + y) / (3 * z)
    assert powsimp((x / 3 + y / 3) / z, deep=True, combine='exp') == (x / 3 + y / 3) / z
    assert powsimp(exp(x) / (1 + exp(x) * exp(y)), deep=True) == exp(x) / (1 + exp(x + y))
    assert powsimp(x * y ** (z ** x * z ** y), deep=True) == x * y ** z ** (x + y)
    assert powsimp((z ** x * z ** y) ** x, deep=True) == (z ** (x + y)) ** x
    assert powsimp(x * (z ** x * z ** y) ** x, deep=True) == x * (z ** (x + y)) ** x
    p = symbols('p', positive=True)
    assert powsimp((1 / x) ** log(2) / x) == (1 / x) ** (1 + log(2))
    assert powsimp((1 / p) ** log(2) / p) == p ** (-1 - log(2))
    assert powsimp(2 ** (2 * x)) == 4 ** x
    assert powsimp((-1) ** (2 * x)) == (-1) ** (2 * x)
    i = symbols('i', integer=True)
    assert powsimp((-1) ** (2 * i)) == 1
    assert powsimp((-1) ** (-x)) != (-1) ** x
    assert powsimp((-1) ** (2 * x), force=True) == 1
    w, n, m = symbols('w n m', negative=True)
    e = i / a
    ex = w ** e * n ** e * m ** e
    assert powsimp(ex) == m ** (i / a) * n ** (i / a) * w ** (i / a)
    e = i / 3
    ex = w ** e * n ** e * m ** e
    assert powsimp(ex) == (-1) ** i * (-m * n * w) ** (i / 3)
    e = (3 + i) / i
    ex = w ** e * n ** e * m ** e
    assert powsimp(ex) == (-1) ** (3 * e) * (-m * n * w) ** e
    eq = x ** (a * Rational(2, 3))
    assert powsimp(eq).exp == eq.exp == a * Rational(2, 3)
    assert powsimp(2 ** (2 * x)) == 4 ** x
    assert powsimp(exp(p / 2)) == exp(p / 2)
    eq = Mul(*[sqrt(Dummy(imaginary=True)) for i in range(3)])
    assert powsimp(eq) == eq and eq.is_Mul
    assert all((powsimp(e) == e for e in (sqrt(x ** a), sqrt(x ** 2))))
    assert str(powsimp(exp(I * pi / 3) * root(-1, 3))) == '(-1)**(2/3)'
    assert powsimp(-0.1 ** x) == -0.1 ** x
    assert powsimp((1 / (2 * E)) ** oo) == (exp(-1) / 2) ** oo
    eq = sin(2 * x) ** 2 * sin(2.0 * x) ** 2
    assert powsimp(eq) == eq
    assert powsimp(x ** 2 * y ** 3 * (x * y ** 2) ** Rational(3, 2)) == x * y * (x * y ** 2) ** Rational(5, 2)