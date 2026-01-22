from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.series.order import O
from sympy.sets.sets import Interval
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import simplify
from sympy.core.exprtools import (decompose_power, Factors, Term, _gcd_terms,
from sympy.core.mul import _keep_coeff as _keep_coeff
from sympy.simplify.cse_opts import sub_pre
from sympy.testing.pytest import raises
from sympy.abc import a, b, t, x, y, z
def test_issue_17256():
    from sympy.sets.fancysets import Range
    x = Symbol('x')
    s1 = Sum(x + 1, (x, 1, 9))
    s2 = Sum(x + 1, (x, Range(1, 10)))
    a = Symbol('a')
    r1 = s1.xreplace({x: a})
    r2 = s2.xreplace({x: a})
    assert r1.doit() == r2.doit()
    s1 = Sum(x + 1, (x, 0, 9))
    s2 = Sum(x + 1, (x, Range(10)))
    a = Symbol('a')
    r1 = s1.xreplace({x: a})
    r2 = s2.xreplace({x: a})
    assert r1 == r2