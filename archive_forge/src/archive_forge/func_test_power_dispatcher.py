from sympy.core import (
from sympy.core.parameters import global_parameters
from sympy.core.tests.test_evalf import NS
from sympy.core.function import expand_multinomial
from sympy.functions.elementary.miscellaneous import sqrt, cbrt
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.special.error_functions import erf
from sympy.functions.elementary.trigonometric import (
from sympy.functions.elementary.hyperbolic import cosh, sinh, tanh
from sympy.polys import Poly
from sympy.series.order import O
from sympy.sets import FiniteSet
from sympy.core.power import power, integer_nthroot
from sympy.testing.pytest import warns, _both_exp_pow
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.abc import a, b, c, x, y
def test_power_dispatcher():

    class NewBase(Expr):
        pass

    class NewPow(NewBase, Pow):
        pass
    a, b = (Symbol('a'), NewBase())

    @power.register(Expr, NewBase)
    @power.register(NewBase, Expr)
    @power.register(NewBase, NewBase)
    def _(a, b):
        return NewPow(a, b)
    assert power(2, 3) == 8 * S.One
    assert power(a, 2) == Pow(a, 2)
    assert power(a, a) == Pow(a, a)
    assert power(a, b) == NewPow(a, b)
    assert power(b, a) == NewPow(b, a)
    assert power(b, b) == NewPow(b, b)