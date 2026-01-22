from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.elementary.complexes import (conjugate, im, re)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, cos, sin)
from sympy.polys.domains.integerring import ZZ
from sympy.sets.sets import Interval
from sympy.simplify.powsimp import powsimp
from sympy.polys import Poly, cyclotomic_poly, intervals, nroots, rootof
from sympy.polys.polyroots import (root_factors, roots_linear,
from sympy.polys.orthopolys import legendre_poly
from sympy.polys.polyerrors import PolynomialError, \
from sympy.polys.polyutils import _nsort
from sympy.testing.pytest import raises, slow
from sympy.core.random import verify_numerically
import mpmath
from itertools import product
def test_roots_mixed():
    f = -1936 - 5056 * x - 7592 * x ** 2 + 2704 * x ** 3 - 49 * x ** 4
    _re, _im = intervals(f, all=True)
    _nroots = nroots(f)
    _sroots = roots(f, multiple=True)
    _re = [Interval(a, b) for (a, b), _ in _re]
    _im = [Interval(re(a), re(b)) * Interval(im(a), im(b)) for (a, b), _ in _im]
    _intervals = _re + _im
    _sroots = [r.evalf() for r in _sroots]
    _nroots = sorted(_nroots, key=lambda x: x.sort_key())
    _sroots = sorted(_sroots, key=lambda x: x.sort_key())
    for _roots in (_nroots, _sroots):
        for i, r in zip(_intervals, _roots):
            if r.is_real:
                assert r in i
            else:
                assert (re(r), im(r)) in i