from sympy.polys.rings import ring, xring
from sympy.polys.domains import FF, ZZ, QQ, ZZ_I, QQ_I, RR, EX
from sympy.polys import polyconfig as config
from sympy.polys.polyerrors import DomainError
from sympy.polys.polyclasses import ANP
from sympy.polys.specialpolys import f_polys, w_polys
from sympy.core.numbers import I
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises, XFAIL
def test_dup_trial_division():
    R, x = ring('x', ZZ)
    assert R.dup_trial_division(x ** 5 + 8 * x ** 4 + 25 * x ** 3 + 38 * x ** 2 + 28 * x + 8, (x + 1, x + 2)) == [(x + 1, 2), (x + 2, 3)]