from sympy.core import S, Function, diff, Tuple, Dummy, Mul
from sympy.core.basic import Basic, as_Basic
from sympy.core.numbers import Rational, NumberSymbol, _illegal
from sympy.core.parameters import global_parameters
from sympy.core.relational import (Lt, Gt, Eq, Ne, Relational,
from sympy.core.sorting import ordered
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.logic.boolalg import (And, Boolean, distribute_and_over_or, Not,
from sympy.utilities.iterables import uniq, sift, common_prefix
from sympy.utilities.misc import filldedent, func_name
from itertools import product
def make_exclusive(*pwargs):
    cumcond = false
    newargs = []
    for expr_i, cond_i in pwargs[:-1]:
        cancond = And(cond_i, Not(cumcond)).simplify()
        cumcond = Or(cond_i, cumcond).simplify()
        newargs.append((expr_i, cancond))
    expr_n, cond_n = pwargs[-1]
    cancond_n = And(cond_n, Not(cumcond)).simplify()
    newargs.append((expr_n, cancond_n))
    if not skip_nan:
        cumcond = Or(cond_n, cumcond).simplify()
        if cumcond is not true:
            newargs.append((Undefined, Not(cumcond).simplify()))
    return Piecewise(*newargs, evaluate=False)