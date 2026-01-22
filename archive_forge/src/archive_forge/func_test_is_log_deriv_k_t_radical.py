from sympy.integrals.risch import DifferentialExtension, derivation
from sympy.integrals.prde import (prde_normal_denom, prde_special_denom,
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.polytools import Poly
from sympy.abc import x, t, n
def test_is_log_deriv_k_t_radical():
    DE = DifferentialExtension(extension={'D': [Poly(1, x)], 'exts': [None], 'extargs': [None]})
    assert is_log_deriv_k_t_radical(Poly(2 * x, x), Poly(1, x), DE) is None
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(2 * t1, t1), Poly(1 / x, t2)], 'exts': [None, 'exp', 'log'], 'extargs': [None, 2 * x, x]})
    assert is_log_deriv_k_t_radical(Poly(x + t2 / 2, t2), Poly(1, t2), DE) == ([(t1, 1), (x, 1)], t1 * x, 2, 0)
    DE = DifferentialExtension(extension={'D': [Poly(1, x), Poly(t0, t0), Poly(1 / x, t)], 'exts': [None, 'exp', 'log'], 'extargs': [None, x, x]})
    assert is_log_deriv_k_t_radical(Poly(x + t / 2 + 3, t), Poly(1, t), DE) == ([(t0, 2), (x, 1)], x * t0 ** 2, 2, 3)