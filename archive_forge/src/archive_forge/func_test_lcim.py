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
def test_lcim():
    assert lcim([S.Half, S(2), S(3)]) == 6
    assert lcim([pi / 2, pi / 4, pi]) == pi
    assert lcim([2 * pi, pi / 2]) == 2 * pi
    assert lcim([S.One, 2 * pi]) is None
    assert lcim([S(2) + 2 * E, E / 3 + Rational(1, 3), S.One + E]) == S(2) + 2 * E