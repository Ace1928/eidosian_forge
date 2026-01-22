from sympy.core.numbers import (oo, pi)
from sympy.core.symbol import Wild
from sympy.core.expr import Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import sin, cos, sinc
from sympy.series.series_class import SeriesBase
from sympy.series.sequences import SeqFormula
from sympy.sets.sets import Interval
from sympy.utilities.iterables import is_sequence
def finite_check(f, x, L):

    def check_fx(exprs, x):
        return x not in exprs.free_symbols

    def check_sincos(_expr, x, L):
        if isinstance(_expr, (sin, cos)):
            sincos_args = _expr.args[0]
            if sincos_args.match(a * (pi / L) * x + b) is not None:
                return True
            else:
                return False
    from sympy.simplify.fu import TR2, TR1, sincos_to_sum
    _expr = sincos_to_sum(TR2(TR1(f)))
    add_coeff = _expr.as_coeff_add()
    a = Wild('a', properties=[lambda k: k.is_Integer, lambda k: k != S.Zero])
    b = Wild('b', properties=[lambda k: x not in k.free_symbols])
    for s in add_coeff[1]:
        mul_coeffs = s.as_coeff_mul()[1]
        for t in mul_coeffs:
            if not (check_fx(t, x) or check_sincos(t, x, L)):
                return (False, f)
    return (True, _expr)