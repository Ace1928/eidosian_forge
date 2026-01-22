from sympy.polys.fields import field, sfield, FracField, FracElement
from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ
from sympy.polys.orderings import lex
from sympy.testing.pytest import raises, XFAIL
from sympy.core import symbols, E
from sympy.core.numbers import Rational
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
def test_FracField___eq__():
    assert field('x,y,z', QQ)[0] == field('x,y,z', QQ)[0]
    assert field('x,y,z', QQ)[0] is field('x,y,z', QQ)[0]
    assert field('x,y,z', QQ)[0] != field('x,y,z', ZZ)[0]
    assert field('x,y,z', QQ)[0] is not field('x,y,z', ZZ)[0]
    assert field('x,y,z', ZZ)[0] != field('x,y,z', QQ)[0]
    assert field('x,y,z', ZZ)[0] is not field('x,y,z', QQ)[0]
    assert field('x,y,z', QQ)[0] != field('x,y', QQ)[0]
    assert field('x,y,z', QQ)[0] is not field('x,y', QQ)[0]
    assert field('x,y', QQ)[0] != field('x,y,z', QQ)[0]
    assert field('x,y', QQ)[0] is not field('x,y,z', QQ)[0]