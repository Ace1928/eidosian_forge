from sympy.core.function import (Derivative, Function, Subs, diff)
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import acosh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan2, cos, sin, tan)
from sympy.integrals.integrals import Integral
from sympy.polys.polytools import Poly
from sympy.series.order import O
from sympy.simplify.radsimp import collect
from sympy.solvers.ode import (classify_ode,
from sympy.solvers.ode.subscheck import checkodesol
from sympy.solvers.ode.ode import (classify_sysode,
from sympy.solvers.ode.nonhomogeneous import _undetermined_coefficients_match
from sympy.solvers.ode.single import LinearCoefficients
from sympy.solvers.deutils import ode_order
from sympy.testing.pytest import XFAIL, raises, slow
from sympy.utilities.misc import filldedent
def test_solve_ics():
    assert dsolve(f(x).diff(x) - 1 / f(x), f(x), ics={f(1): 2}) == Eq(f(x), sqrt(2 * x + 2))
    assert dsolve(f(x).diff(x) - f(x), f(x), ics={f(0): 1}) == Eq(f(x), exp(x))
    assert dsolve(f(x).diff(x) - f(x), f(x), ics={f(x).diff(x).subs(x, 0): 1}) == Eq(f(x), exp(x))
    assert dsolve(f(x).diff(x, x) + f(x), f(x), ics={f(0): 1, f(x).diff(x).subs(x, 0): 1}) == Eq(f(x), sin(x) + cos(x))
    assert dsolve([f(x).diff(x) - f(x) + g(x), g(x).diff(x) - g(x) - f(x)], [f(x), g(x)], ics={f(0): 1, g(0): 0}) == [Eq(f(x), exp(x) * cos(x)), Eq(g(x), exp(x) * sin(x))]
    eq = (x ** 2 * f(x) ** 2 - x).diff(x)
    assert dsolve(eq, f(x), ics={f(1): 0}) == [Eq(f(x), -sqrt(x - 1) / x), Eq(f(x), sqrt(x - 1) / x)]
    assert dsolve(eq, f(x), ics={f(x).diff(x).subs(x, 1): 0}) == [Eq(f(x), -sqrt(x - S.Half) / x), Eq(f(x), sqrt(x - S.Half) / x)]
    eq = cos(f(x)) - (x * sin(f(x)) - f(x) ** 2) * f(x).diff(x)
    assert dsolve(eq, f(x), ics={f(0): 1}, hint='1st_exact', simplify=False) == Eq(x * cos(f(x)) + f(x) ** 3 / 3, Rational(1, 3))
    assert dsolve(eq, f(x), ics={f(0): 1}, hint='1st_exact', simplify=True) == Eq(x * cos(f(x)) + f(x) ** 3 / 3, Rational(1, 3))
    assert solve_ics([Eq(f(x), C1 * exp(x))], [f(x)], [C1], {f(0): 1}) == {C1: 1}
    assert solve_ics([Eq(f(x), C1 * sin(x) + C2 * cos(x))], [f(x)], [C1, C2], {f(0): 1, f(pi / 2): 1}) == {C1: 1, C2: 1}
    assert solve_ics([Eq(f(x), C1 * sin(x) + C2 * cos(x))], [f(x)], [C1, C2], {f(0): 1, f(x).diff(x).subs(x, 0): 1}) == {C1: 1, C2: 1}
    assert solve_ics([Eq(f(x), C1 * sin(x) + C2 * cos(x))], [f(x)], [C1, C2], {f(0): 1}) == {C2: 1}
    assert set(dsolve(f(x).diff(x) * (f(x).diff(x, 2) - x), ics={f(0): 0, f(x).diff(x).subs(x, 1): 0})) == {Eq(f(x), 0), Eq(f(x), x ** 3 / 6 - x / 2)}
    assert set(dsolve(f(x).diff(x) * (f(x).diff(x, 2) - x), ics={f(0): 0})) == {Eq(f(x), 0), Eq(f(x), C2 * x + x ** 3 / 6)}
    K, r, f0 = symbols('K r f0')
    sol = Eq(f(x), K * f0 * exp(r * x) / ((-K + f0) * (f0 * exp(r * x) / (-K + f0) - 1)))
    assert dsolve(Eq(f(x).diff(x), r * f(x) * (1 - f(x) / K)), f(x), ics={f(0): f0}) == sol
    assert set(dsolve(f(x).diff(x) * (f(x).diff(x, 2) - x), ics={f(x).diff(x).subs(x, 0): 0, f(0): 0})) == {Eq(f(x), 0), Eq(f(x), x ** 3 / 6)}
    assert set(dsolve(f(x).diff(x) * (f(x).diff(x, 2) - x), ics={f(0): 0, f(x).diff(x).subs(x, 0): 0})) == {Eq(f(x), 0), Eq(f(x), x ** 3 / 6)}
    raises(ValueError, lambda: solve_ics([Eq(f(x), C1 * sin(x) + C2 * cos(x))], [f(x)], [C1, C2], {f(0): 1, f(pi): 1}))
    raises(ValueError, lambda: solve_ics([Eq(f(x), sqrt(C1 - x ** 2))], [f(x)], [C1], {f(x).diff(x).subs(x, 0): 0}))
    EI, q, L = symbols('EI q L')
    sols = [Eq(f(x), C1 + C2 * x + C3 * x ** 2 + C4 * x ** 3 + q * x ** 4 / (24 * EI))]
    funcs = [f(x)]
    constants = [C1, C2, C3, C4]
    ics1 = {f(0): 0, f(x).diff(x).subs(x, 0): 0, f(L).diff(L, 2): 0, f(L).diff(L, 3): 0}
    ics2 = {f(0): 0, f(x).diff(x).subs(x, 0): 0, Subs(f(x).diff(x, 2), x, L): 0, Subs(f(x).diff(x, 3), x, L): 0}
    solved_constants1 = solve_ics(sols, funcs, constants, ics1)
    solved_constants2 = solve_ics(sols, funcs, constants, ics2)
    assert solved_constants1 == solved_constants2 == {C1: 0, C2: 0, C3: L ** 2 * q / (4 * EI), C4: -L * q / (6 * EI)}
    ics = {f(0): f(0)}
    assert dsolve(f(x).diff(x) - f(x), f(x), ics=ics) == Eq(f(x), f(0) * exp(x))
    ics = {f(x).diff(x).subs(x, 0): f(x).diff(x).subs(x, 0), f(0): f(0)}
    assert dsolve(f(x).diff(x, x) + f(x), f(x), ics=ics) == Eq(f(x), f(0) * cos(x) + f(x).diff(x).subs(x, 0) * sin(x))