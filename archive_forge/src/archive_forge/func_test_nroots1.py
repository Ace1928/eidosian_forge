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
@slow
def test_nroots1():
    n = 64
    p = legendre_poly(n, x, polys=True)
    raises(mpmath.mp.NoConvergence, lambda: p.nroots(n=3, maxsteps=5))
    roots = p.nroots(n=3)
    assert [str(r) for r in roots] == ['-0.999', '-0.996', '-0.991', '-0.983', '-0.973', '-0.961', '-0.946', '-0.930', '-0.911', '-0.889', '-0.866', '-0.841', '-0.813', '-0.784', '-0.753', '-0.720', '-0.685', '-0.649', '-0.611', '-0.572', '-0.531', '-0.489', '-0.446', '-0.402', '-0.357', '-0.311', '-0.265', '-0.217', '-0.170', '-0.121', '-0.0730', '-0.0243', '0.0243', '0.0730', '0.121', '0.170', '0.217', '0.265', '0.311', '0.357', '0.402', '0.446', '0.489', '0.531', '0.572', '0.611', '0.649', '0.685', '0.720', '0.753', '0.784', '0.813', '0.841', '0.866', '0.889', '0.911', '0.930', '0.946', '0.961', '0.973', '0.983', '0.991', '0.996', '0.999']