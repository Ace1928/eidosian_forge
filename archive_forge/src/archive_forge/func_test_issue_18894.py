from functools import reduce
from operator import add, mul
from sympy.polys.rings import ring, xring, sring, PolyRing, PolyElement
from sympy.polys.fields import field, FracField
from sympy.polys.domains import ZZ, QQ, RR, FF, EX
from sympy.polys.orderings import lex, grlex
from sympy.polys.polyerrors import GeneratorsError, \
from sympy.testing.pytest import raises
from sympy.core import Symbol, symbols
from sympy.core.singleton import S
from sympy.core.numbers import (oo, pi)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
def test_issue_18894():
    items = [S(3) / 16 + sqrt(3 * sqrt(3) + 10) / 8, S(1) / 8 + 3 * sqrt(3) / 16, S(1) / 8 + 3 * sqrt(3) / 16, -S(3) / 16 + sqrt(3 * sqrt(3) + 10) / 8]
    R, a = sring(items, extension=True)
    assert R.domain == QQ.algebraic_field(sqrt(3) + sqrt(3 * sqrt(3) + 10))
    assert R.gens == ()
    result = []
    for item in items:
        result.append(R.domain.from_sympy(item))
    assert a == result