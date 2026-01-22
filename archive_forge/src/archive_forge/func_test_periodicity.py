from sympy.core.numbers import (E, I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (Abs, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (cos, cot, csc, sec, sin, tan)
from sympy.functions.special.error_functions import expint
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.simplify.simplify import simplify
from sympy.calculus.util import (function_range, continuous_domain, not_empty_in,
from sympy.sets.sets import (Interval, FiniteSet, Complement, Union)
from sympy.testing.pytest import raises, _both_exp_pow
from sympy.abc import x
@_both_exp_pow
def test_periodicity():
    x = Symbol('x')
    y = Symbol('y')
    z = Symbol('z', real=True)
    assert periodicity(sin(2 * x), x) == pi
    assert periodicity(-2 * tan(4 * x), x) == pi / 4
    assert periodicity(sin(x) ** 2, x) == 2 * pi
    assert periodicity(3 ** tan(3 * x), x) == pi / 3
    assert periodicity(tan(x) * cos(x), x) == 2 * pi
    assert periodicity(sin(x) ** tan(x), x) == 2 * pi
    assert periodicity(tan(x) * sec(x), x) == 2 * pi
    assert periodicity(sin(2 * x) * cos(2 * x) - y, x) == pi / 2
    assert periodicity(tan(x) + cot(x), x) == pi
    assert periodicity(sin(x) - cos(2 * x), x) == 2 * pi
    assert periodicity(sin(x) - 1, x) == 2 * pi
    assert periodicity(sin(4 * x) + sin(x) * cos(x), x) == pi
    assert periodicity(exp(sin(x)), x) == 2 * pi
    assert periodicity(log(cot(2 * x)) - sin(cos(2 * x)), x) == pi
    assert periodicity(sin(2 * x) * exp(tan(x) - csc(2 * x)), x) == pi
    assert periodicity(cos(sec(x) - csc(2 * x)), x) == 2 * pi
    assert periodicity(tan(sin(2 * x)), x) == pi
    assert periodicity(2 * tan(x) ** 2, x) == pi
    assert periodicity(sin(x % 4), x) == 4
    assert periodicity(sin(x) % 4, x) == 2 * pi
    assert periodicity(tan((3 * x - 2) % 4), x) == Rational(4, 3)
    assert periodicity((sqrt(2) * (x + 1) + x) % 3, x) == 3 / (sqrt(2) + 1)
    assert periodicity((x ** 2 + 1) % x, x) is None
    assert periodicity(sin(re(x)), x) == 2 * pi
    assert periodicity(sin(x) ** 2 + cos(x) ** 2, x) is S.Zero
    assert periodicity(tan(x), y) is S.Zero
    assert periodicity(sin(x) + I * cos(x), x) == 2 * pi
    assert periodicity(x - sin(2 * y), y) == pi
    assert periodicity(exp(x), x) is None
    assert periodicity(exp(I * x), x) == 2 * pi
    assert periodicity(exp(I * z), z) == 2 * pi
    assert periodicity(exp(z), z) is None
    assert periodicity(exp(log(sin(z) + I * cos(2 * z)), evaluate=False), z) == 2 * pi
    assert periodicity(exp(log(sin(2 * z) + I * cos(z)), evaluate=False), z) == 2 * pi
    assert periodicity(exp(sin(z)), z) == 2 * pi
    assert periodicity(exp(2 * I * z), z) == pi
    assert periodicity(exp(z + I * sin(z)), z) is None
    assert periodicity(exp(cos(z / 2) + sin(z)), z) == 4 * pi
    assert periodicity(log(x), x) is None
    assert periodicity(exp(x) ** sin(x), x) is None
    assert periodicity(sin(x) ** y, y) is None
    assert periodicity(Abs(sin(Abs(sin(x)))), x) == pi
    assert all((periodicity(Abs(f(x)), x) == pi for f in (cos, sin, sec, csc, tan, cot)))
    assert periodicity(Abs(sin(tan(x))), x) == pi
    assert periodicity(Abs(sin(sin(x) + tan(x))), x) == 2 * pi
    assert periodicity(sin(x) > S.Half, x) == 2 * pi
    assert periodicity(x > 2, x) is None
    assert periodicity(x ** 3 - x ** 2 + 1, x) is None
    assert periodicity(Abs(x), x) is None
    assert periodicity(Abs(x ** 2 - 1), x) is None
    assert periodicity((x ** 2 + 4) % 2, x) is None
    assert periodicity(E ** x % 3, x) is None
    assert periodicity(sin(expint(1, x)) / expint(1, x), x) is None
    p = Piecewise((0, x < -1), (x ** 2, x <= 1), (log(x), True))
    assert periodicity(p, x) is None
    m = MatrixSymbol('m', 3, 3)
    raises(NotImplementedError, lambda: periodicity(sin(m), m))
    raises(NotImplementedError, lambda: periodicity(sin(m[0, 0]), m))
    raises(NotImplementedError, lambda: periodicity(sin(m), m[0, 0]))
    raises(NotImplementedError, lambda: periodicity(sin(m[0, 0]), m[0, 0]))