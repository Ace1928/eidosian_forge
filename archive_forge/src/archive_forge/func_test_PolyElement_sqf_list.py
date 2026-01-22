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
def test_PolyElement_sqf_list():
    _, x = ring('x', ZZ)
    f = x ** 5 - x ** 3 - x ** 2 + 1
    g = x ** 3 + 2 * x ** 2 + 2 * x + 1
    h = x - 1
    p = x ** 4 + x ** 3 - x - 1
    assert f.sqf_part() == p
    assert f.sqf_list() == (1, [(g, 1), (h, 2)])