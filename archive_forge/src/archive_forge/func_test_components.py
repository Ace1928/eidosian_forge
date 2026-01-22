from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Ne
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, cosh, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sin, tan)
from sympy.functions.special.bessel import (besselj, besselk, bessely, jn)
from sympy.functions.special.error_functions import erf
from sympy.integrals.integrals import Integral
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify
from sympy.integrals.heurisch import components, heurisch, heurisch_wrapper
from sympy.testing.pytest import XFAIL, skip, slow, ON_CI
from sympy.integrals.integrals import integrate
def test_components():
    assert components(x * y, x) == {x}
    assert components(1 / (x + y), x) == {x}
    assert components(sin(x), x) == {sin(x), x}
    assert components(sin(x) * sqrt(log(x)), x) == {log(x), sin(x), sqrt(log(x)), x}
    assert components(x * sin(exp(x) * y), x) == {sin(y * exp(x)), x, exp(x)}
    assert components(x ** Rational(17, 54) / sqrt(sin(x)), x) == {sin(x), x ** Rational(1, 54), sqrt(sin(x)), x}
    assert components(f(x), x) == {x, f(x)}
    assert components(Derivative(f(x), x), x) == {x, f(x), Derivative(f(x), x)}
    assert components(f(x) * diff(f(x), x), x) == {x, f(x), Derivative(f(x), x), Derivative(f(x), x)}