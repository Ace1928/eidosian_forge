from sympy.core.containers import Tuple
from sympy.core.function import Derivative
from sympy.core.numbers import (I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import (appellf1, hyper, meijerg)
from sympy.series.order import O
from sympy.abc import x, z, k
from sympy.series.limits import limit
from sympy.testing.pytest import raises, slow
from sympy.core.random import (
def test_hyper_rewrite_sum():
    from sympy.concrete.summations import Sum
    from sympy.core.symbol import Dummy
    from sympy.functions.combinatorial.factorials import RisingFactorial, factorial
    _k = Dummy('k')
    assert replace_dummy(hyper((1, 2), (1, 3), x).rewrite(Sum), _k) == Sum(x ** _k / factorial(_k) * RisingFactorial(2, _k) / RisingFactorial(3, _k), (_k, 0, oo))
    assert hyper((1, 2, 3), (-1, 3), z).rewrite(Sum) == hyper((1, 2, 3), (-1, 3), z)