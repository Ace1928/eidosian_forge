from sympy.calculus.accumulationbounds import AccumBounds
from sympy.core.add import Add
from sympy.core.function import (Lambda, diff)
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import (E, Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (arg, conjugate, im, re)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.hyperbolic import (acoth, asinh, atanh, cosh, coth, sinh, tanh)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, acot, acsc, asec, asin, atan, atan2,
from sympy.functions.special.bessel import (besselj, jn)
from sympy.functions.special.delta_functions import Heaviside
from sympy.matrices.dense import Matrix
from sympy.polys.polytools import (cancel, gcd)
from sympy.series.limits import limit
from sympy.series.order import O
from sympy.series.series import series
from sympy.sets.fancysets import ImageSet
from sympy.sets.sets import (FiniteSet, Interval)
from sympy.simplify.simplify import simplify
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.core.relational import Ne, Eq
from sympy.functions.elementary.piecewise import Piecewise
from sympy.sets.setexpr import SetExpr
from sympy.testing.pytest import XFAIL, slow, raises
@slow
def test_cot_rewrite_slow():
    assert cot(pi * Rational(4, 34)).rewrite(pow).ratsimp() == (cos(pi * Rational(4, 34)) / sin(pi * Rational(4, 34))).rewrite(pow).ratsimp()
    assert cot(pi * Rational(4, 17)).rewrite(pow) == (cos(pi * Rational(4, 17)) / sin(pi * Rational(4, 17))).rewrite(pow)
    assert cot(pi / 19).rewrite(pow) == cot(pi / 19)
    assert cot(pi / 19).rewrite(sqrt) == cot(pi / 19)
    assert cot(pi * Rational(2, 5), evaluate=False).rewrite(sqrt) == (Rational(-1, 4) + sqrt(5) / 4) / sqrt(sqrt(5) / 8 + Rational(5, 8))