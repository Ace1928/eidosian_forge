from sympy.core.function import expand_complex
from sympy.core.numbers import (I, Integer, Rational, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (Abs, conjugate, im, re, sign)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, cot, sin, tan)
def test_real_imag():
    x, y, z = symbols('x, y, z')
    X, Y, Z = symbols('X, Y, Z', commutative=False)
    a = Symbol('a', real=True)
    assert (2 * a * x).as_real_imag() == (2 * a * re(x), 2 * a * im(x))
    assert (x * x.conjugate()).as_real_imag() == (Abs(x) ** 2, 0)
    assert im(x * x.conjugate()) == 0
    assert im(x * y.conjugate() * z * y) == im(x * z) * Abs(y) ** 2
    assert im(x * y.conjugate() * x * y) == im(x ** 2) * Abs(y) ** 2
    assert im(Z * y.conjugate() * X * y) == im(Z * X) * Abs(y) ** 2
    assert im(X * X.conjugate()) == im(X * X.conjugate(), evaluate=False)
    assert (sin(x) * sin(x).conjugate()).as_real_imag() == (Abs(sin(x)) ** 2, 0)
    assert (x ** 2).as_real_imag() == (re(x) ** 2 - im(x) ** 2, 2 * re(x) * im(x))
    r = Symbol('r', real=True)
    i = Symbol('i', imaginary=True)
    assert (i * r * x).as_real_imag() == (I * i * r * im(x), -I * i * r * re(x))
    assert (i * r * x * (y + 2)).as_real_imag() == (I * i * r * (re(y) + 2) * im(x) + I * i * r * re(x) * im(y), -I * i * r * (re(y) + 2) * re(x) + I * i * r * im(x) * im(y))
    assert ((1 + I) / (1 - I)).as_real_imag() == (0, 1)
    assert ((1 + 2 * I) * (1 + 3 * I)).as_real_imag() == (-5, 5)