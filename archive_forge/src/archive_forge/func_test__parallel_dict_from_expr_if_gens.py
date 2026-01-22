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
def test__parallel_dict_from_expr_if_gens():
    assert parallel_dict_from_expr([x + 2 * y + 3 * z, Integer(7)], gens=(x,)) == ([{(1,): Integer(1), (0,): 2 * y + 3 * z}, {(0,): Integer(7)}], (x,))