from itertools import islice
from sympy.core import Add, S, Mul, Pow
from sympy.core.exprtools import factor_terms
from sympy.core.function import Function, AppliedUndef, expand
from sympy.core.relational import Equality, Eq
from sympy.core.symbol import Symbol, Wild, Dummy, symbols
from sympy.functions import exp, log
from sympy.integrals.integrals import integrate
from sympy.polys import Poly
from sympy.polys.polytools import cancel, div
from sympy.simplify import (collect, powsimp,  # type: ignore
from sympy.solvers import solve
from sympy.solvers.pde import pdsolve
from sympy.utilities import numbered_symbols
from sympy.solvers.deutils import _preprocess, ode_order
from .ode import checkinfsol
def lie_heuristic_abaco1_product(match, comp=False):
    """
    The second heuristic uses the following two assumptions on `\\xi` and `\\eta`

    .. math:: \\eta = 0, \\xi = f(x)*g(y)

    .. math:: \\eta = f(x)*g(y), \\xi = 0

    The first assumption of this heuristic holds good if
    `\\frac{1}{h^{2}}\\frac{\\partial^2}{\\partial x \\partial y}\\log(h)` is
    separable in `x` and `y`, then the separated factors containing `x`
    is `f(x)`, and `g(y)` is obtained by

    .. math:: e^{\\int f\\frac{\\partial}{\\partial x}\\left(\\frac{1}{f*h}\\right)\\,dy}

    provided `f\\frac{\\partial}{\\partial x}\\left(\\frac{1}{f*h}\\right)` is a function
    of `y` only.

    The second assumption holds good if `\\frac{dy}{dx} = h(x, y)` is rewritten as
    `\\frac{dy}{dx} = \\frac{1}{h(y, x)}` and the same properties of the first assumption
    satisfies. After obtaining `f(x)` and `g(y)`, the coordinates are again
    interchanged, to get `\\eta` as `f(x)*g(y)`


    References
    ==========
    - E.S. Cheb-Terrab, A.D. Roche, Symmetries and First Order
      ODE Patterns, pp. 7 - pp. 8

    """
    xieta = []
    y = match['y']
    h = match['h']
    hinv = match['hinv']
    func = match['func']
    x = func.args[0]
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)
    inf = separatevars(log(h).diff(y).diff(x) / h ** 2, dict=True, symbols=[x, y])
    if inf and inf['coeff']:
        fx = inf[x]
        gy = simplify(fx * (1 / (fx * h)).diff(x))
        gysyms = gy.free_symbols
        if x not in gysyms:
            gy = exp(integrate(gy, y))
            inf = {eta: S.Zero, xi: (fx * gy).subs(y, func)}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)
    u1 = Dummy('u1')
    inf = separatevars(log(hinv).diff(y).diff(x) / hinv ** 2, dict=True, symbols=[x, y])
    if inf and inf['coeff']:
        fx = inf[x]
        gy = simplify(fx * (1 / (fx * hinv)).diff(x))
        gysyms = gy.free_symbols
        if x not in gysyms:
            gy = exp(integrate(gy, y))
            etaval = fx * gy
            etaval = etaval.subs([(x, u1), (y, x)]).subs(u1, y)
            inf = {eta: etaval.subs(y, func), xi: S.Zero}
            if not comp:
                return [inf]
            if comp and inf not in xieta:
                xieta.append(inf)
    if xieta:
        return xieta