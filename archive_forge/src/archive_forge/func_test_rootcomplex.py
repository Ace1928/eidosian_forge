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
def test_rootcomplex():
    R = Rational
    assert ((+1 + I) ** R(1, 2)).expand(complex=True) == 2 ** R(1, 4) * cos(pi / 8) + 2 ** R(1, 4) * sin(pi / 8) * I
    assert ((-1 - I) ** R(1, 2)).expand(complex=True) == 2 ** R(1, 4) * cos(3 * pi / 8) - 2 ** R(1, 4) * sin(3 * pi / 8) * I
    assert (sqrt(-10) * I).as_real_imag() == (-sqrt(10), 0)