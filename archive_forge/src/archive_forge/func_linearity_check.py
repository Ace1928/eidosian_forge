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
def linearity_check(eqs, j, func, is_linear_):
    for k in range(order[func] + 1):
        func_coef[j, func, k] = collect(eqs.expand(), [diff(func, t, k)]).coeff(diff(func, t, k))
        if is_linear_ == True:
            if func_coef[j, func, k] == 0:
                if k == 0:
                    coef = eqs.as_independent(func, as_Add=True)[1]
                    for xr in range(1, ode_order(eqs, func) + 1):
                        coef -= eqs.as_independent(diff(func, t, xr), as_Add=True)[1]
                    if coef != 0:
                        is_linear_ = False
                elif eqs.as_independent(diff(func, t, k), as_Add=True)[1]:
                    is_linear_ = False
            else:
                for func_ in funcs:
                    if isinstance(func_, list):
                        for elem_func_ in func_:
                            dep = func_coef[j, func, k].as_independent(elem_func_, as_Add=True)[1]
                            if dep != 0:
                                is_linear_ = False
                    else:
                        dep = func_coef[j, func, k].as_independent(func_, as_Add=True)[1]
                        if dep != 0:
                            is_linear_ = False
    return is_linear_