from sympy.core import Add, S, Mul, Pow, oo
from sympy.core.containers import Tuple
from sympy.core.expr import AtomicExpr, Expr
from sympy.core.function import (Function, Derivative, AppliedUndef, diff,
from sympy.core.multidimensional import vectorize
from sympy.core.numbers import nan, zoo, Number
from sympy.core.relational import Equality, Eq
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, Wild, Dummy, symbols
from sympy.core.sympify import sympify
from sympy.core.traversal import preorder_traversal
from sympy.logic.boolalg import (BooleanAtom, BooleanTrue,
from sympy.functions import exp, log, sqrt
from sympy.functions.combinatorial.factorials import factorial
from sympy.integrals.integrals import Integral
from sympy.polys import (Poly, terms_gcd, PolynomialError, lcm)
from sympy.polys.polytools import cancel
from sympy.series import Order
from sympy.series.series import series
from sympy.simplify import (collect, logcombine, powsimp,  # type: ignore
from sympy.simplify.radsimp import collect_const
from sympy.solvers import checksol, solve
from sympy.utilities import numbered_symbols
from sympy.utilities.iterables import uniq, sift, iterable
from sympy.solvers.deutils import _preprocess, ode_order, _desolve
from .single import SingleODEProblem, SingleODESolver, solver_map
def check_linear_2eq_order1(eq, func, func_coef):
    x = func[0].func
    y = func[1].func
    fc = func_coef
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    r = {}
    r['a1'] = fc[0, x(t), 1]
    r['a2'] = fc[1, y(t), 1]
    r['b1'] = -fc[0, x(t), 0] / fc[0, x(t), 1]
    r['b2'] = -fc[1, x(t), 0] / fc[1, y(t), 1]
    r['c1'] = -fc[0, y(t), 0] / fc[0, x(t), 1]
    r['c2'] = -fc[1, y(t), 0] / fc[1, y(t), 1]
    forcing = [S.Zero, S.Zero]
    for i in range(2):
        for j in Add.make_args(eq[i]):
            if not j.has(x(t), y(t)):
                forcing[i] += j
    if not (forcing[0].has(t) or forcing[1].has(t)):
        r['d1'] = forcing[0]
        r['d2'] = forcing[1]
    else:
        return None
    p = 0
    q = 0
    p1 = cancel(r['b2'] / cancel(r['b2'] / r['c2']).as_numer_denom()[0])
    p2 = cancel(r['b1'] / cancel(r['b1'] / r['c1']).as_numer_denom()[0])
    for n, i in enumerate([p1, p2]):
        for j in Mul.make_args(collect_const(i)):
            if not j.has(t):
                q = j
            if q and n == 0:
                if (r['b2'] / j - r['b1']) / (r['c1'] - r['c2'] / j) == j:
                    p = 1
            elif q and n == 1:
                if (r['b1'] / j - r['b2']) / (r['c2'] - r['c1'] / j) == j:
                    p = 2
    if r['d1'] != 0 or r['d2'] != 0:
        return None
    elif not any((r[k].has(t) for k in 'a1 a2 b1 b2 c1 c2'.split())):
        return None
    else:
        r['b1'] = r['b1'] / r['a1']
        r['b2'] = r['b2'] / r['a2']
        r['c1'] = r['c1'] / r['a1']
        r['c2'] = r['c2'] / r['a2']
        if p:
            return 'type6'
        else:
            return 'type7'