from inspect import isroutine
from pyomo.core import Var, Objective, Constraint, Set, Param
def process_canonical_repn(expr):
    """
    Returns a dictionary of {var_name_or_None: coef} values. None
    indicates a numeric constant.
    """
    terms = {}
    vars = expr.pop(-1, {})
    linear = expr.pop(1, {})
    for k in linear:
        v = linear[k]
        terms[vars[k.keys()[0]].label] = v
    const = expr.pop(0, {})
    if None in const:
        terms[None] = const[None]
    if len(expr) != 0:
        raise TypeError('Nonlinear terms in expression')
    return terms