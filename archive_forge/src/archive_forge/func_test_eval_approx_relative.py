from sympy.polys.polytools import Poly
import sympy.polys.rootoftools as rootoftools
from sympy.polys.rootoftools import (rootof, RootOf, CRootOf, RootSum,
from sympy.polys.polyerrors import (
from sympy.core.function import (Function, Lambda)
from sympy.core.numbers import (Float, I, Rational)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import tan
from sympy.integrals.integrals import Integral
from sympy.polys.orthopolys import legendre_poly
from sympy.solvers.solvers import solve
from sympy.testing.pytest import raises, slow
from sympy.core.expr import unchanged
from sympy.abc import a, b, x, y, z, r
@slow
def test_eval_approx_relative():
    CRootOf.clear_cache()
    t = [CRootOf(x ** 3 + 10 * x + 1, i) for i in range(3)]
    assert [i.eval_rational(0.1) for i in t] == [Rational(-21, 220), Rational(15, 256) - I * 805 / 256, Rational(15, 256) + I * 805 / 256]
    t[0]._reset()
    assert [i.eval_rational(0.1, 0.0001) for i in t] == [Rational(-21, 220), Rational(3275, 65536) - I * 414645 / 131072, Rational(3275, 65536) + I * 414645 / 131072]
    assert S(t[0]._get_interval().dx) < 0.1
    assert S(t[1]._get_interval().dx) < 0.1
    assert S(t[1]._get_interval().dy) < 0.0001
    assert S(t[2]._get_interval().dx) < 0.1
    assert S(t[2]._get_interval().dy) < 0.0001
    t[0]._reset()
    assert [i.eval_rational(0.0001, 0.0001) for i in t] == [Rational(-2001, 20020), Rational(6545, 131072) - I * 414645 / 131072, Rational(6545, 131072) + I * 414645 / 131072]
    assert S(t[0]._get_interval().dx) < 0.0001
    assert S(t[1]._get_interval().dx) < 0.0001
    assert S(t[1]._get_interval().dy) < 0.0001
    assert S(t[2]._get_interval().dx) < 0.0001
    assert S(t[2]._get_interval().dy) < 0.0001
    t[0]._reset()
    assert [i.eval_rational(n=2) for i in t] == [Rational(-202201, 2024022), Rational(104755, 2097152) - I * 6634255 / 2097152, Rational(104755, 2097152) + I * 6634255 / 2097152]
    assert abs(S(t[0]._get_interval().dx) / t[0]) < 0.01
    assert abs(S(t[1]._get_interval().dx) / t[1]).n() < 0.01
    assert abs(S(t[1]._get_interval().dy) / t[1]).n() < 0.01
    assert abs(S(t[2]._get_interval().dx) / t[2]).n() < 0.01
    assert abs(S(t[2]._get_interval().dy) / t[2]).n() < 0.01
    t[0]._reset()
    assert [i.eval_rational(n=3) for i in t] == [Rational(-202201, 2024022), Rational(1676045, 33554432) - I * 106148135 / 33554432, Rational(1676045, 33554432) + I * 106148135 / 33554432]
    assert abs(S(t[0]._get_interval().dx) / t[0]) < 0.001
    assert abs(S(t[1]._get_interval().dx) / t[1]).n() < 0.001
    assert abs(S(t[1]._get_interval().dy) / t[1]).n() < 0.001
    assert abs(S(t[2]._get_interval().dx) / t[2]).n() < 0.001
    assert abs(S(t[2]._get_interval().dy) / t[2]).n() < 0.001
    t[0]._reset()
    a = [i.eval_approx(2) for i in t]
    assert [str(i) for i in a] == ['-0.10', '0.05 - 3.2*I', '0.05 + 3.2*I']
    assert all((abs(((a[i] - t[i]) / t[i]).n()) < 0.01 for i in range(len(a))))