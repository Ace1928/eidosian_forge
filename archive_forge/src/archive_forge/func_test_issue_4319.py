from sympy import abc
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.hyper import meijerg
from sympy.polys.polytools import Poly
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import signsimp
from sympy.testing.pytest import XFAIL
def test_issue_4319():
    x, y = symbols('x y')
    p = -x * (S.One / 8 - y)
    ans = {S.Zero, y - S.One / 8}

    def ok(pat):
        assert set(p.match(pat).values()) == ans
    ok(Wild('coeff', exclude=[x]) * x + Wild('rest'))
    ok(Wild('w', exclude=[x]) * x + Wild('rest'))
    ok(Wild('coeff', exclude=[x]) * x + Wild('rest'))
    ok(Wild('w', exclude=[x]) * x + Wild('rest'))
    ok(Wild('e', exclude=[x]) * x + Wild('rest'))
    ok(Wild('ress', exclude=[x]) * x + Wild('rest'))
    ok(Wild('resu', exclude=[x]) * x + Wild('rest'))