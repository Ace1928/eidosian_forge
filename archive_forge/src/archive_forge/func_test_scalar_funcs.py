from sympy.sets.setexpr import SetExpr
from sympy.sets import Interval, FiniteSet, Intersection, ImageSet, Union
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.trigonometric import cos
from sympy.sets.sets import Set
def test_scalar_funcs():
    assert SetExpr(Interval(0, 1)).set == Interval(0, 1)
    a, b = (Symbol('a', real=True), Symbol('b', real=True))
    a, b = (1, 2)
    for f in [exp, log]:
        input_se = f(SetExpr(Interval(a, b)))
        output = input_se.set
        expected = Interval(Min(f(a), f(b)), Max(f(a), f(b)))
        assert output == expected