import pickle
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.external import import_module
from sympy.testing.pytest import skip
from sympy.utilities.matchpy_connector import WildDot, WildPlus, WildStar, Replacer
def test_replacer():
    if matchpy is None:
        skip('matchpy not installed')
    x1_ = WildDot('x1_')
    x2_ = WildDot('x2_')
    a_ = WildDot('a_', optional=S.One)
    b_ = WildDot('b_', optional=S.One)
    c_ = WildDot('c_', optional=S.Zero)
    replacer = Replacer(common_constraints=[matchpy.CustomConstraint(lambda a_: not a_.has(x)), matchpy.CustomConstraint(lambda b_: not b_.has(x)), matchpy.CustomConstraint(lambda c_: not c_.has(x))])
    replacer.add(Eq(x1_, x2_), Eq(x1_ - x2_, 0), conditions_nonfalse=[Ne(x2_, 0), Ne(x1_, 0), Ne(x1_, x), Ne(x2_, x)])
    replacer.add(Eq(a_ * x + b_, 0), Eq(x, -b_ / a_))
    disc = b_ ** 2 - 4 * a_ * c_
    replacer.add(Eq(a_ * x ** 2 + b_ * x + c_, 0), Eq(x, (-b_ - sqrt(disc)) / (2 * a_)) | Eq(x, (-b_ + sqrt(disc)) / (2 * a_)), conditions_nonfalse=[disc >= 0])
    replacer.add(Eq(a_ * x ** 2 + c_, 0), Eq(x, sqrt(-c_ / a_)) | Eq(x, -sqrt(-c_ / a_)), conditions_nonfalse=[-c_ * a_ > 0])
    assert replacer.replace(Eq(3 * x, y)) == Eq(x, y / 3)
    assert replacer.replace(Eq(x ** 2 + 1, 0)) == Eq(x ** 2 + 1, 0)
    assert replacer.replace(Eq(x ** 2, 4)) == Eq(x, 2) | Eq(x, -2)
    assert replacer.replace(Eq(x ** 2 + 4 * y * x + 4 * y ** 2, 0)) == Eq(x, -2 * y)