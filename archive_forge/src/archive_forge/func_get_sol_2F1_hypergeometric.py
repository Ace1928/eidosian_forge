from sympy.core import S, Pow
from sympy.core.function import expand
from sympy.core.relational import Eq
from sympy.core.symbol import Symbol, Wild
from sympy.functions import exp, sqrt, hyper
from sympy.integrals import Integral
from sympy.polys import roots, gcd
from sympy.polys.polytools import cancel, factor
from sympy.simplify import collect, simplify, logcombine # type: ignore
from sympy.simplify.powsimp import powdenest
from sympy.solvers.ode.ode import get_numbered_constants
def get_sol_2F1_hypergeometric(eq, func, match_object):
    x = func.args[0]
    from sympy.simplify.hyperexpand import hyperexpand
    from sympy.polys.polytools import factor
    C0, C1 = get_numbered_constants(eq, num=2)
    a = match_object['a']
    b = match_object['b']
    c = match_object['c']
    A = match_object['A']
    sol = None
    if c.is_integer == False:
        sol = C0 * hyper([a, b], [c], x) + C1 * hyper([a - c + 1, b - c + 1], [2 - c], x) * x ** (1 - c)
    elif c == 1:
        y2 = Integral(exp(Integral((-(a + b + 1) * x + c) / (x ** 2 - x), x)) / hyperexpand(hyper([a, b], [c], x)) ** 2, x) * hyper([a, b], [c], x)
        sol = C0 * hyper([a, b], [c], x) + C1 * y2
    elif (c - a - b).is_integer == False:
        sol = C0 * hyper([a, b], [1 + a + b - c], 1 - x) + C1 * hyper([c - a, c - b], [1 + c - a - b], 1 - x) * (1 - x) ** (c - a - b)
    if sol:
        subs = match_object['mobius']
        dtdx = simplify(1 / subs.diff(x))
        _B = ((a + b + 1) * x - c).subs(x, subs) * dtdx
        _B = factor(_B + (x ** 2 - x).subs(x, subs) * (dtdx.diff(x) * dtdx))
        _A = factor((x ** 2 - x).subs(x, subs) * dtdx ** 2)
        e = exp(logcombine(Integral(cancel(_B / (2 * _A)), x), force=True))
        sol = sol.subs(x, match_object['mobius'])
        sol = sol.subs(x, x ** match_object['k'])
        e = e.subs(x, x ** match_object['k'])
        if not A.is_zero:
            e1 = Integral(A / 2, x)
            e1 = exp(logcombine(e1, force=True))
            sol = cancel(e / e1 * x ** ((-match_object['k'] + 1) / 2)) * sol
            sol = Eq(func, sol)
            return sol
        sol = cancel(e * x ** ((-match_object['k'] + 1) / 2)) * sol
        sol = Eq(func, sol)
    return sol