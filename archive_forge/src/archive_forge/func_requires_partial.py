import re
from collections.abc import Iterable
from sympy.core.function import Derivative
def requires_partial(expr):
    """Return whether a partial derivative symbol is required for printing

    This requires checking how many free variables there are,
    filtering out the ones that are integers. Some expressions do not have
    free variables. In that case, check its variable list explicitly to
    get the context of the expression.
    """
    if isinstance(expr, Derivative):
        return requires_partial(expr.expr)
    if not isinstance(expr.free_symbols, Iterable):
        return len(set(expr.variables)) > 1
    return sum((not s.is_integer for s in expr.free_symbols)) > 1