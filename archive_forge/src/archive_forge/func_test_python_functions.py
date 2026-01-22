from sympy.core.function import (Derivative, Function)
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (Abs, conjugate)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.integrals.integrals import Integral
from sympy.matrices.dense import Matrix
from sympy.series.limits import limit
from sympy.printing.python import python
from sympy.testing.pytest import raises, XFAIL
def test_python_functions():
    assert python(2 * x + exp(x)) in "x = Symbol('x')\ne = 2*x + exp(x)"
    assert python(sqrt(2)) == 'e = sqrt(2)'
    assert python(2 ** Rational(1, 3)) == 'e = 2**Rational(1, 3)'
    assert python(sqrt(2 + pi)) == 'e = sqrt(2 + pi)'
    assert python((2 + pi) ** Rational(1, 3)) == 'e = (2 + pi)**Rational(1, 3)'
    assert python(2 ** Rational(1, 4)) == 'e = 2**Rational(1, 4)'
    assert python(Abs(x)) == "x = Symbol('x')\ne = Abs(x)"
    assert python(Abs(x / (x ** 2 + 1))) in ["x = Symbol('x')\ne = Abs(x/(1 + x**2))", "x = Symbol('x')\ne = Abs(x/(x**2 + 1))"]
    f = Function('f')
    assert python(f(x)) == "x = Symbol('x')\nf = Function('f')\ne = f(x)"
    assert python(f(x, y)) == "x = Symbol('x')\ny = Symbol('y')\nf = Function('f')\ne = f(x, y)"
    assert python(f(x / (y + 1), y)) in ["x = Symbol('x')\ny = Symbol('y')\nf = Function('f')\ne = f(x/(1 + y), y)", "x = Symbol('x')\ny = Symbol('y')\nf = Function('f')\ne = f(x/(y + 1), y)"]
    assert python(sqrt(sqrt(x + 1) + 1)) in ["x = Symbol('x')\ne = sqrt(1 + sqrt(1 + x))", "x = Symbol('x')\ne = sqrt(sqrt(x + 1) + 1)"]
    assert python(((x + 1) ** Rational(1, 3) + 1) ** Rational(1, 3)) in ["x = Symbol('x')\ne = (1 + (1 + x)**Rational(1, 3))**Rational(1, 3)", "x = Symbol('x')\ne = ((x + 1)**Rational(1, 3) + 1)**Rational(1, 3)"]
    assert python(sin(x) ** 2) == "x = Symbol('x')\ne = sin(x)**2"