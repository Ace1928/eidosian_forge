from sympy.core.containers import Tuple
from sympy.core.numbers import (AlgebraicNumber, I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys.polytools import Poly
from sympy.polys.numberfields.subfield import to_number_field
from sympy.polys.polyclasses import DMP
from sympy.polys.domains import QQ
from sympy.polys.rootoftools import CRootOf
from sympy.abc import x, y
def test_to_algebraic_integer():
    a = AlgebraicNumber(sqrt(3), gen=x).to_algebraic_integer()
    assert a.minpoly == x ** 2 - 3
    assert a.root == sqrt(3)
    assert a.rep == DMP([QQ(1), QQ(0)], QQ)
    a = AlgebraicNumber(2 * sqrt(3), gen=x).to_algebraic_integer()
    assert a.minpoly == x ** 2 - 12
    assert a.root == 2 * sqrt(3)
    assert a.rep == DMP([QQ(1), QQ(0)], QQ)
    a = AlgebraicNumber(sqrt(3) / 2, gen=x).to_algebraic_integer()
    assert a.minpoly == x ** 2 - 12
    assert a.root == 2 * sqrt(3)
    assert a.rep == DMP([QQ(1), QQ(0)], QQ)
    a = AlgebraicNumber(sqrt(3) / 2, [Rational(7, 19), 3], gen=x).to_algebraic_integer()
    assert a.minpoly == x ** 2 - 12
    assert a.root == 2 * sqrt(3)
    assert a.rep == DMP([QQ(7, 19), QQ(3)], QQ)