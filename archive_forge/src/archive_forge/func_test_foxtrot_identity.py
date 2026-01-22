from mpmath import *
from mpmath.libmp import round_up, from_float, mpf_zeta_int
def test_foxtrot_identity():
    psi0 = lambda z: psi(0, z)
    mp.dps = 50
    a = (-1) ** fraction(1, 3)
    b = (-1) ** fraction(2, 3)
    x = -psi0(0.5 * a) - psi0(-0.5 * b) + psi0(0.5 * (1 + a)) + psi0(0.5 * (1 - b))
    y = 2 * pi * sech(0.5 * sqrt(3) * pi)
    assert x.ae(y)
    mp.dps = 15