from sympy.core.function import (Derivative, Function, diff)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.error_functions import (Ei, erf, erfi)
from sympy.integrals.integrals import Integral
from sympy.solvers.ode.subscheck import checkodesol, checksysodesol
from sympy.functions import besselj, bessely
from sympy.testing.pytest import raises, slow
@slow
def test_checkodesol():
    raises(ValueError, lambda: checkodesol(f(x, y).diff(x), Eq(f(x, y), x)))
    raises(ValueError, lambda: checkodesol(f(x).diff(x), Eq(f(x, y), x), f(x, y)))
    assert checkodesol(f(x).diff(x), Eq(f(x, y), x)) == (False, -f(x).diff(x) + f(x, y).diff(x) - 1)
    assert checkodesol(f(x).diff(x), Eq(f(x), x)) is not True
    assert checkodesol(f(x).diff(x), Eq(f(x), x)) == (False, 1)
    sol1 = Eq(f(x) ** 5 + 11 * f(x) - 2 * f(x) + x, 0)
    assert checkodesol(diff(sol1.lhs, x), sol1) == (True, 0)
    assert checkodesol(diff(sol1.lhs, x) * exp(f(x)), sol1) == (True, 0)
    assert checkodesol(diff(sol1.lhs, x, 2), sol1) == (True, 0)
    assert checkodesol(diff(sol1.lhs, x, 2) * exp(f(x)), sol1) == (True, 0)
    assert checkodesol(diff(sol1.lhs, x, 3), sol1) == (True, 0)
    assert checkodesol(diff(sol1.lhs, x, 3) * exp(f(x)), sol1) == (True, 0)
    assert checkodesol(diff(sol1.lhs, x, 3), Eq(f(x), x * log(x))) == (False, 60 * x ** 4 * ((log(x) + 1) ** 2 + log(x)) * (log(x) + 1) * log(x) ** 2 - 5 * x ** 4 * log(x) ** 4 - 9)
    assert checkodesol(diff(exp(f(x)) + x, x) * x, Eq(exp(f(x)) + x, 0)) == (True, 0)
    assert checkodesol(diff(exp(f(x)) + x, x) * x, Eq(exp(f(x)) + x, 0), solve_for_func=False) == (True, 0)
    assert checkodesol(f(x).diff(x, 2), [Eq(f(x), C1 + C2 * x), Eq(f(x), C2 + C1 * x), Eq(f(x), C1 * x + C2 * x ** 2)]) == [(True, 0), (True, 0), (False, C2)]
    assert checkodesol(f(x).diff(x, 2), {Eq(f(x), C1 + C2 * x), Eq(f(x), C2 + C1 * x), Eq(f(x), C1 * x + C2 * x ** 2)}) == {(True, 0), (True, 0), (False, C2)}
    assert checkodesol(f(x).diff(x) - 1 / f(x) / 2, Eq(f(x) ** 2, x)) == [(True, 0), (True, 0)]
    assert checkodesol(f(x).diff(x) - f(x), Eq(C1 * exp(x), f(x))) == (True, 0)
    eq3 = x * exp(f(x) / x) + f(x) - x * f(x).diff(x)
    sol3 = Eq(f(x), log(log(C1 / x) ** (-x)))
    assert not checkodesol(eq3, sol3)[1].has(f(x))
    eqn = Eq(Derivative(x * Derivative(f(x), x), x) / x, exp(x))
    sol = Eq(f(x), C1 + C2 * log(x) + exp(x) - Ei(x))
    assert checkodesol(eqn, sol, order=2, solve_for_func=False)[0]
    eq = x ** 2 * f(x).diff(x, 2) + x * f(x).diff(x) + (2 * x ** 2 + 25) * f(x)
    sol = Eq(f(x), C1 * besselj(5 * I, sqrt(2) * x) + C2 * bessely(5 * I, sqrt(2) * x))
    assert checkodesol(eq, sol) == (True, 0)
    eqs = [Eq(f(x).diff(x), f(x) + g(x)), Eq(g(x).diff(x), f(x) + g(x))]
    sol = [Eq(f(x), -C1 + C2 * exp(2 * x)), Eq(g(x), C1 + C2 * exp(2 * x))]
    assert checkodesol(eqs, sol) == (True, [0, 0])