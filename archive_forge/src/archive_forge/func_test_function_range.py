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
def test_function_range():
    x, y, a, b = symbols('x y a b')
    assert function_range(sin(x), x, Interval(-pi / 2, pi / 2)) == Interval(-1, 1)
    assert function_range(sin(x), x, Interval(0, pi)) == Interval(0, 1)
    assert function_range(tan(x), x, Interval(0, pi)) == Interval(-oo, oo)
    assert function_range(tan(x), x, Interval(pi / 2, pi)) == Interval(-oo, 0)
    assert function_range((x + 3) / (x - 2), x, Interval(-5, 5)) == Union(Interval(-oo, Rational(2, 7)), Interval(Rational(8, 3), oo))
    assert function_range(1 / x ** 2, x, Interval(-1, 1)) == Interval(1, oo)
    assert function_range(exp(x), x, Interval(-1, 1)) == Interval(exp(-1), exp(1))
    assert function_range(log(x) - x, x, S.Reals) == Interval(-oo, -1)
    assert function_range(sqrt(3 * x - 1), x, Interval(0, 2)) == Interval(0, sqrt(5))
    assert function_range(x * (x - 1) - (x ** 2 - x), x, S.Reals) == FiniteSet(0)
    assert function_range(x * (x - 1) - (x ** 2 - x) + y, x, S.Reals) == FiniteSet(y)
    assert function_range(sin(x), x, Union(Interval(-5, -3), FiniteSet(4))) == Union(Interval(-sin(3), 1), FiniteSet(sin(4)))
    assert function_range(cos(x), x, Interval(-oo, -4)) == Interval(-1, 1)
    assert function_range(cos(x), x, S.EmptySet) == S.EmptySet
    assert function_range(x / sqrt(x ** 2 + 1), x, S.Reals) == Interval.open(-1, 1)
    raises(NotImplementedError, lambda: function_range(exp(x) * (sin(x) - cos(x)) / 2 - x, x, S.Reals))
    raises(NotImplementedError, lambda: function_range(sin(x) + x, x, S.Reals))
    raises(NotImplementedError, lambda: function_range(log(x), x, S.Integers))
    raises(NotImplementedError, lambda: function_range(sin(x) / 2, x, S.Naturals))