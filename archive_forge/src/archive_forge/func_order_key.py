from sympy.core import (S, Add, Mul, Pow, Eq, Expr,
from sympy.core.exprtools import decompose_power, decompose_power_rat
from sympy.core.numbers import _illegal
from sympy.polys.polyerrors import PolynomialError, GeneratorsError
from sympy.polys.polyoptions import build_options
import re
def order_key(gen):
    gen = str(gen)
    if wrt is not None:
        try:
            return (-len(wrt) + wrt.index(gen), gen, 0)
        except ValueError:
            pass
    name, index = _re_gen.match(gen).groups()
    if index:
        index = int(index)
    else:
        index = 0
    try:
        return (gens_order[name], name, index)
    except KeyError:
        pass
    try:
        return (_gens_order[name], name, index)
    except KeyError:
        pass
    return (_max_order, name, index)