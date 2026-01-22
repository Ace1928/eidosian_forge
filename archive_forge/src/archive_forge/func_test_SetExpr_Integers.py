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
def test_SetExpr_Integers():
    assert SetExpr(S.Integers) + 1 == SetExpr(S.Integers)
    assert (SetExpr(S.Integers) + I).dummy_eq(SetExpr(ImageSet(Lambda(_d, _d + I), S.Integers)))
    assert SetExpr(S.Integers) * -1 == SetExpr(S.Integers)
    assert (SetExpr(S.Integers) * 2).dummy_eq(SetExpr(ImageSet(Lambda(_d, 2 * _d), S.Integers)))
    assert (SetExpr(S.Integers) * I).dummy_eq(SetExpr(ImageSet(Lambda(_d, I * _d), S.Integers)))
    assert SetExpr(S.Integers)._eval_func(Lambda(x, I * x + 1)).dummy_eq(SetExpr(ImageSet(Lambda(_d, I * _d + 1), S.Integers)))
    assert (SetExpr(S.Integers) * I + 1).dummy_eq(SetExpr(ImageSet(Lambda(x, x + 1), ImageSet(Lambda(_d, _d * I), S.Integers))))