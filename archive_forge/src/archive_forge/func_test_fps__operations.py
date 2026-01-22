from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acosh, asech)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin)
from sympy.functions.special.bessel import airyai
from sympy.functions.special.error_functions import erf
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import integrate
from sympy.series.formal import fps
from sympy.series.order import O
from sympy.series.formal import (rational_algorithm, FormalPowerSeries,
from sympy.testing.pytest import raises, XFAIL, slow
def test_fps__operations():
    f1, f2 = (fps(sin(x)), fps(cos(x)))
    fsum = f1 + f2
    assert fsum.function == sin(x) + cos(x)
    assert fsum.truncate() == 1 + x - x ** 2 / 2 - x ** 3 / 6 + x ** 4 / 24 + x ** 5 / 120 + O(x ** 6)
    fsum = f1 + 1
    assert fsum.function == sin(x) + 1
    assert fsum.truncate() == 1 + x - x ** 3 / 6 + x ** 5 / 120 + O(x ** 6)
    fsum = 1 + f2
    assert fsum.function == cos(x) + 1
    assert fsum.truncate() == 2 - x ** 2 / 2 + x ** 4 / 24 + O(x ** 6)
    assert f1 + x == Add(f1, x)
    assert -f2.truncate() == -1 + x ** 2 / 2 - x ** 4 / 24 + O(x ** 6)
    assert f1 - f1 is S.Zero
    fsub = f1 - f2
    assert fsub.function == sin(x) - cos(x)
    assert fsub.truncate() == -1 + x + x ** 2 / 2 - x ** 3 / 6 - x ** 4 / 24 + x ** 5 / 120 + O(x ** 6)
    fsub = f1 - 1
    assert fsub.function == sin(x) - 1
    assert fsub.truncate() == -1 + x - x ** 3 / 6 + x ** 5 / 120 + O(x ** 6)
    fsub = 1 - f2
    assert fsub.function == -cos(x) + 1
    assert fsub.truncate() == x ** 2 / 2 - x ** 4 / 24 + O(x ** 6)
    raises(ValueError, lambda: f1 + fps(exp(x), dir=-1))
    raises(ValueError, lambda: f1 + fps(exp(x), x0=1))
    fm = f1 * 3
    assert fm.function == 3 * sin(x)
    assert fm.truncate() == 3 * x - x ** 3 / 2 + x ** 5 / 40 + O(x ** 6)
    fm = 3 * f2
    assert fm.function == 3 * cos(x)
    assert fm.truncate() == 3 - 3 * x ** 2 / 2 + x ** 4 / 8 + O(x ** 6)
    assert f1 * f2 == Mul(f1, f2)
    assert f1 * x == Mul(f1, x)
    fd = f1.diff()
    assert fd.function == cos(x)
    assert fd.truncate() == 1 - x ** 2 / 2 + x ** 4 / 24 + O(x ** 6)
    fd = f2.diff()
    assert fd.function == -sin(x)
    assert fd.truncate() == -x + x ** 3 / 6 - x ** 5 / 120 + O(x ** 6)
    fd = f2.diff().diff()
    assert fd.function == -cos(x)
    assert fd.truncate() == -1 + x ** 2 / 2 - x ** 4 / 24 + O(x ** 6)
    f3 = fps(exp(sqrt(x)))
    fd = f3.diff()
    assert fd.truncate().expand() == 1 / (2 * sqrt(x)) + S.Half + x / 12 + x ** 2 / 240 + x ** 3 / 10080 + x ** 4 / 725760 + x ** 5 / 79833600 + sqrt(x) / 4 + x ** Rational(3, 2) / 48 + x ** Rational(5, 2) / 1440 + x ** Rational(7, 2) / 80640 + x ** Rational(9, 2) / 7257600 + x ** Rational(11, 2) / 958003200 + O(x ** 6)
    assert f1.integrate((x, 0, 1)) == -cos(1) + 1
    assert integrate(f1, (x, 0, 1)) == -cos(1) + 1
    fi = integrate(f1, x)
    assert fi.function == -cos(x)
    assert fi.truncate() == -1 + x ** 2 / 2 - x ** 4 / 24 + O(x ** 6)
    fi = f2.integrate(x)
    assert fi.function == sin(x)
    assert fi.truncate() == x - x ** 3 / 6 + x ** 5 / 120 + O(x ** 6)