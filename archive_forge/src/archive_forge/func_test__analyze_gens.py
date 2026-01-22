from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.testing.pytest import raises
from sympy.polys.polyutils import (
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.domains import ZZ
def test__analyze_gens():
    assert _analyze_gens((x, y, z)) == (x, y, z)
    assert _analyze_gens([x, y, z]) == (x, y, z)
    assert _analyze_gens(([x, y, z],)) == (x, y, z)
    assert _analyze_gens(((x, y, z),)) == (x, y, z)