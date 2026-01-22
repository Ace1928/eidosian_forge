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
def test_python_integrals():
    f_1 = Integral(log(x), x)
    assert python(f_1) == "x = Symbol('x')\ne = Integral(log(x), x)"
    f_2 = Integral(x ** 2, x)
    assert python(f_2) == "x = Symbol('x')\ne = Integral(x**2, x)"
    f_3 = Integral(x ** 2 ** x, x)
    assert python(f_3) == "x = Symbol('x')\ne = Integral(x**(2**x), x)"
    f_4 = Integral(x ** 2, (x, 1, 2))
    assert python(f_4) == "x = Symbol('x')\ne = Integral(x**2, (x, 1, 2))"
    f_5 = Integral(x ** 2, (x, Rational(1, 2), 10))
    assert python(f_5) == "x = Symbol('x')\ne = Integral(x**2, (x, Rational(1, 2), 10))"
    f_6 = Integral(x ** 2 * y ** 2, x, y)
    assert python(f_6) == "x = Symbol('x')\ny = Symbol('y')\ne = Integral(x**2*y**2, x, y)"