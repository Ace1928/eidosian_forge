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
def test_dup_zz_hensel_step():
    R, x = ring('x', ZZ)
    f = x ** 4 - 1
    g = x ** 3 + 2 * x ** 2 - x - 2
    h = x - 2
    s = -2
    t = 2 * x ** 2 - 2 * x - 1
    G, H, S, T = R.dup_zz_hensel_step(5, f, g, h, s, t)
    assert G == x ** 3 + 7 * x ** 2 - x - 7
    assert H == x - 7
    assert S == 8
    assert T == -8 * x ** 2 - 12 * x - 1