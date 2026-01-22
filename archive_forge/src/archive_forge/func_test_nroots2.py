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
def test_nroots2():
    p = Poly(x ** 5 + 3 * x + 1, x)
    roots = p.nroots(n=3)
    assert [str(r) for r in roots] == ['-0.332', '-0.839 - 0.944*I', '-0.839 + 0.944*I', '1.01 - 0.937*I', '1.01 + 0.937*I']
    roots = p.nroots(n=5)
    assert [str(r) for r in roots] == ['-0.33199', '-0.83907 - 0.94385*I', '-0.83907 + 0.94385*I', '1.0051 - 0.93726*I', '1.0051 + 0.93726*I']