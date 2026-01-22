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
def lie_heuristic_chi(match, comp=False):
    """
    The aim of the fourth heuristic is to find the function `\\chi(x, y)`
    that satisfies the PDE `\\frac{d\\chi}{dx} + h\\frac{d\\chi}{dx}
    - \\frac{\\partial h}{\\partial y}\\chi = 0`.

    This assumes `\\chi` to be a bivariate polynomial in `x` and `y`. By intuition,
    `h` should be a rational function in `x` and `y`. The method used here is
    to substitute a general binomial for `\\chi` up to a certain maximum degree
    is reached. The coefficients of the polynomials, are calculated by by collecting
    terms of the same order in `x` and `y`.

    After finding `\\chi`, the next step is to use `\\eta = \\xi*h + \\chi`, to
    determine `\\xi` and `\\eta`. This can be done by dividing `\\chi` by `h`
    which would give `-\\xi` as the quotient and `\\eta` as the remainder.


    References
    ==========
    - E.S Cheb-Terrab, L.G.S Duarte and L.A,C.P da Mota, Computer Algebra
      Solving of First Order ODEs Using Symmetry Methods, pp. 8

    """
    h = match['h']
    hy = match['hy']
    func = match['func']
    x = func.args[0]
    y = match['y']
    xi = Function('xi')(x, func)
    eta = Function('eta')(x, func)
    if h.is_rational_function():
        schi, schix, schiy = symbols('schi, schix, schiy')
        cpde = schix + h * schiy - hy * schi
        num, denom = cancel(cpde).as_numer_denom()
        deg = Poly(num, x, y).total_degree()
        chi = Function('chi')(x, y)
        chix = chi.diff(x)
        chiy = chi.diff(y)
        cpde = chix + h * chiy - hy * chi
        chieq = Symbol('chi')
        for i in range(1, deg + 1):
            chieq += Add(*[Symbol('chi_' + str(power) + '_' + str(i - power)) * x ** power * y ** (i - power) for power in range(i + 1)])
            cnum, cden = cancel(cpde.subs({chi: chieq}).doit()).as_numer_denom()
            cnum = expand(cnum)
            if cnum.is_polynomial(x, y) and cnum.is_Add:
                cpoly = Poly(cnum, x, y).as_dict()
                if cpoly:
                    solsyms = chieq.free_symbols - {x, y}
                    soldict = solve(cpoly.values(), *solsyms)
                    if isinstance(soldict, list):
                        soldict = soldict[0]
                    if any(soldict.values()):
                        chieq = chieq.subs(soldict)
                        dict_ = {sym: 1 for sym in solsyms}
                        chieq = chieq.subs(dict_)
                        xic, etac = div(chieq, h)
                        inf = {eta: etac.subs(y, func), xi: -xic.subs(y, func)}
                        return [inf]