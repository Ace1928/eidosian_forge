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
def test_parallel_dict_from_expr():
    assert parallel_dict_from_expr([Eq(x, 1), Eq(x ** 2, 2)]) == ([{(0,): -Integer(1), (1,): Integer(1)}, {(0,): -Integer(2), (2,): Integer(1)}], (x,))
    raises(PolynomialError, lambda: parallel_dict_from_expr([A * B - B * A]))