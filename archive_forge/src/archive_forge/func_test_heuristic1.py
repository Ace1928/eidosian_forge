from sympy.core.function import Function
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan, sin, tan)
from sympy.solvers.ode import (classify_ode, checkinfsol, dsolve, infinitesimals)
from sympy.solvers.ode.subscheck import checkodesol
from sympy.testing.pytest import XFAIL
def test_heuristic1():
    a, b, c, a4, a3, a2, a1, a0 = symbols('a b c a4 a3 a2 a1 a0')
    df = f(x).diff(x)
    eq = Eq(df, x ** 2 * f(x))
    eq1 = f(x).diff(x) + a * f(x) - c * exp(b * x)
    eq2 = f(x).diff(x) + 2 * x * f(x) - x * exp(-x ** 2)
    eq3 = (1 + 2 * x) * df + 2 - 4 * exp(-f(x))
    eq4 = f(x).diff(x) - (a4 * x ** 4 + a3 * x ** 3 + a2 * x ** 2 + a1 * x + a0) ** Rational(-1, 2)
    eq5 = x ** 2 * df - f(x) + x ** 2 * exp(x - 1 / x)
    eqlist = [eq, eq1, eq2, eq3, eq4, eq5]
    i = infinitesimals(eq, hint='abaco1_simple')
    assert i == [{eta(x, f(x)): exp(x ** 3 / 3), xi(x, f(x)): 0}, {eta(x, f(x)): f(x), xi(x, f(x)): 0}, {eta(x, f(x)): 0, xi(x, f(x)): x ** (-2)}]
    i1 = infinitesimals(eq1, hint='abaco1_simple')
    assert i1 == [{eta(x, f(x)): exp(-a * x), xi(x, f(x)): 0}]
    i2 = infinitesimals(eq2, hint='abaco1_simple')
    assert i2 == [{eta(x, f(x)): exp(-x ** 2), xi(x, f(x)): 0}]
    i3 = infinitesimals(eq3, hint='abaco1_simple')
    assert i3 == [{eta(x, f(x)): 0, xi(x, f(x)): 2 * x + 1}, {eta(x, f(x)): 0, xi(x, f(x)): 1 / (exp(f(x)) - 2)}]
    i4 = infinitesimals(eq4, hint='abaco1_simple')
    assert i4 == [{eta(x, f(x)): 1, xi(x, f(x)): 0}, {eta(x, f(x)): 0, xi(x, f(x)): sqrt(a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3 + a4 * x ** 4)}]
    i5 = infinitesimals(eq5, hint='abaco1_simple')
    assert i5 == [{xi(x, f(x)): 0, eta(x, f(x)): exp(-1 / x)}]
    ilist = [i, i1, i2, i3, i4, i5]
    for eq, i in zip(eqlist, ilist):
        check = checkinfsol(eq, i)
        assert check[0]
    eq6 = df - f(x) / x * (x * log(x ** 2 / f(x)) + 2)
    i = infinitesimals(eq6, hint='abaco1_product')
    assert i == [{eta(x, f(x)): f(x) * exp(-x), xi(x, f(x)): 0}]
    assert checkinfsol(eq6, i)[0]
    eq7 = x * f(x).diff(x) + 1 - f(x) ** 2
    i = infinitesimals(eq7, hint='chi')
    assert checkinfsol(eq7, i)[0]