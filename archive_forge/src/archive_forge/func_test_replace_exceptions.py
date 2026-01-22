import collections
from sympy.assumptions.ask import Q
from sympy.core.basic import (Basic, Atom, as_Basic,
from sympy.core.containers import Tuple
from sympy.core.function import Function, Lambda
from sympy.core.numbers import I, pi
from sympy.core.singleton import S
from sympy.core.symbol import symbols, Symbol, Dummy
from sympy.concrete.summations import Sum
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.functions.elementary.exponential import exp
from sympy.testing.pytest import raises, warns_deprecated_sympy
def test_replace_exceptions():
    from sympy.core.symbol import Wild
    x, y = symbols('x y')
    e = x ** 2 + x * y
    raises(TypeError, lambda: e.replace(sin, 2))
    b = Wild('b')
    c = Wild('c')
    raises(TypeError, lambda: e.replace(b * c, c.is_real))
    raises(TypeError, lambda: e.replace(b.is_real, 1))
    raises(TypeError, lambda: e.replace(lambda d: d.is_Number, 1))