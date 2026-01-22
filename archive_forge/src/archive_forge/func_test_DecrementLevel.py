from sympy.core.function import (Function, Lambda, diff, expand_log)
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan, sin, tan)
from sympy.polys.polytools import (Poly, cancel, factor)
from sympy.integrals.risch import (gcdex_diophantine, frac_in, as_poly_1t,
from sympy.testing.pytest import raises
from sympy.abc import x, t, nu, z, a, y
def test_DecrementLevel():
    DE = DifferentialExtension(x * log(exp(x) + 1), x)
    assert DE.level == -1
    assert DE.t == t1
    assert DE.d == Poly(t0 / (t0 + 1), t1)
    assert DE.case == 'primitive'
    with DecrementLevel(DE):
        assert DE.level == -2
        assert DE.t == t0
        assert DE.d == Poly(t0, t0)
        assert DE.case == 'exp'
        with DecrementLevel(DE):
            assert DE.level == -3
            assert DE.t == x
            assert DE.d == Poly(1, x)
            assert DE.case == 'base'
        assert DE.level == -2
        assert DE.t == t0
        assert DE.d == Poly(t0, t0)
        assert DE.case == 'exp'
    assert DE.level == -1
    assert DE.t == t1
    assert DE.d == Poly(t0 / (t0 + 1), t1)
    assert DE.case == 'primitive'
    try:
        with DecrementLevel(DE):
            raise _TestingException
    except _TestingException:
        pass
    else:
        raise AssertionError('Did not raise.')
    assert DE.level == -1
    assert DE.t == t1
    assert DE.d == Poly(t0 / (t0 + 1), t1)
    assert DE.case == 'primitive'