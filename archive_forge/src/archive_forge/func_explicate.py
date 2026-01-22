from sympy.core import Function, S, Mul, Pow, Add
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.function import expand_func
from sympy.core.symbol import Dummy
from sympy.functions import gamma, sqrt, sin
from sympy.polys import factor, cancel
from sympy.utilities.iterables import sift, uniq
def explicate(p):
    if p is S.One:
        return (None, [])
    b, e = p.as_base_exp()
    if e.is_Integer:
        if isinstance(b, gamma):
            return (True, [b.args[0]] * e)
        else:
            return (False, [b] * e)
    else:
        return (False, [p])