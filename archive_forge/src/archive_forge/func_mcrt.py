from sympy.ntheory.modular import crt, crt1, crt2, solve_congruence
from sympy.testing.pytest import raises
def mcrt(m, v, r, symmetric=False):
    assert crt(m, v, symmetric)[0] == r
    mm, e, s = crt1(m)
    assert crt2(m, v, mm, e, s, symmetric) == (r, mm)