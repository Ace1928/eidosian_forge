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
def test__parallel_dict_from_expr_no_gens():
    assert parallel_dict_from_expr([x * y, Integer(3)]) == ([{(1, 1): Integer(1)}, {(0, 0): Integer(3)}], (x, y))
    assert parallel_dict_from_expr([x * y, 2 * z, Integer(3)]) == ([{(1, 1, 0): Integer(1)}, {(0, 0, 1): Integer(2)}, {(0, 0, 0): Integer(3)}], (x, y, z))
    assert parallel_dict_from_expr((Mul(x, x ** 2, evaluate=False),)) == ([{(3,): 1}], (x,))