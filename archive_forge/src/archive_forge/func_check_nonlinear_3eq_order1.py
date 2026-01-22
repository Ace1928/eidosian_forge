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
def check_nonlinear_3eq_order1(eq, func, func_coef):
    x = func[0].func
    y = func[1].func
    z = func[2].func
    fc = func_coef
    t = list(list(eq[0].atoms(Derivative))[0].atoms(Symbol))[0]
    u, v, w = symbols('u, v, w', cls=Dummy)
    a = Wild('a', exclude=[x(t), y(t), z(t), t])
    b = Wild('b', exclude=[x(t), y(t), z(t), t])
    c = Wild('c', exclude=[x(t), y(t), z(t), t])
    f = Wild('f')
    F1 = Wild('F1')
    F2 = Wild('F2')
    F3 = Wild('F3')
    for i in range(3):
        eqs = 0
        for terms in Add.make_args(eq[i]):
            eqs += terms / fc[i, func[i], 1]
        eq[i] = eqs
    r1 = eq[0].match(diff(x(t), t) - a * y(t) * z(t))
    r2 = eq[1].match(diff(y(t), t) - b * z(t) * x(t))
    r3 = eq[2].match(diff(z(t), t) - c * x(t) * y(t))
    if r1 and r2 and r3:
        num1, den1 = r1[a].as_numer_denom()
        num2, den2 = r2[b].as_numer_denom()
        num3, den3 = r3[c].as_numer_denom()
        if solve([num1 * u - den1 * (v - w), num2 * v - den2 * (w - u), num3 * w - den3 * (u - v)], [u, v]):
            return 'type1'
    r = eq[0].match(diff(x(t), t) - y(t) * z(t) * f)
    if r:
        r1 = collect_const(r[f]).match(a * f)
        r2 = ((diff(y(t), t) - eq[1]) / r1[f]).match(b * z(t) * x(t))
        r3 = ((diff(z(t), t) - eq[2]) / r1[f]).match(c * x(t) * y(t))
    if r1 and r2 and r3:
        num1, den1 = r1[a].as_numer_denom()
        num2, den2 = r2[b].as_numer_denom()
        num3, den3 = r3[c].as_numer_denom()
        if solve([num1 * u - den1 * (v - w), num2 * v - den2 * (w - u), num3 * w - den3 * (u - v)], [u, v]):
            return 'type2'
    r = eq[0].match(diff(x(t), t) - (F2 - F3))
    if r:
        r1 = collect_const(r[F2]).match(c * F2)
        r1.update(collect_const(r[F3]).match(b * F3))
        if r1:
            if eq[1].has(r1[F2]) and (not eq[1].has(r1[F3])):
                r1[F2], r1[F3] = (r1[F3], r1[F2])
                r1[c], r1[b] = (-r1[b], -r1[c])
            r2 = eq[1].match(diff(y(t), t) - a * r1[F3] + r1[c] * F1)
        if r2:
            r3 = eq[2] == diff(z(t), t) - r1[b] * r2[F1] + r2[a] * r1[F2]
        if r1 and r2 and r3:
            return 'type3'
    r = eq[0].match(diff(x(t), t) - z(t) * F2 + y(t) * F3)
    if r:
        r1 = collect_const(r[F2]).match(c * F2)
        r1.update(collect_const(r[F3]).match(b * F3))
        if r1:
            if eq[1].has(r1[F2]) and (not eq[1].has(r1[F3])):
                r1[F2], r1[F3] = (r1[F3], r1[F2])
                r1[c], r1[b] = (-r1[b], -r1[c])
            r2 = (diff(y(t), t) - eq[1]).match(a * x(t) * r1[F3] - r1[c] * z(t) * F1)
        if r2:
            r3 = diff(z(t), t) - eq[2] == r1[b] * y(t) * r2[F1] - r2[a] * x(t) * r1[F2]
        if r1 and r2 and r3:
            return 'type4'
    r = (diff(x(t), t) - eq[0]).match(x(t) * (F2 - F3))
    if r:
        r1 = collect_const(r[F2]).match(c * F2)
        r1.update(collect_const(r[F3]).match(b * F3))
        if r1:
            if eq[1].has(r1[F2]) and (not eq[1].has(r1[F3])):
                r1[F2], r1[F3] = (r1[F3], r1[F2])
                r1[c], r1[b] = (-r1[b], -r1[c])
            r2 = (diff(y(t), t) - eq[1]).match(y(t) * (a * r1[F3] - r1[c] * F1))
        if r2:
            r3 = diff(z(t), t) - eq[2] == z(t) * (r1[b] * r2[F1] - r2[a] * r1[F2])
        if r1 and r2 and r3:
            return 'type5'
    return None