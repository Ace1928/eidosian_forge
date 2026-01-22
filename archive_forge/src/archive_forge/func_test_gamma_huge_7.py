from mpmath import *
from mpmath.libmp import round_up, from_float, mpf_zeta_int
def test_gamma_huge_7():
    mp.dps = 100
    a = 3 + j / mpf(10) ** 1000
    mp.dps = 15
    y = gamma(a)
    assert str(y.real) == '2.0'
    assert str(y.imag) == '1.84556867019693e-1000'
    mp.dps = 50
    y = gamma(a)
    assert str(y.real) == '2.0'
    assert str(y.imag) == '1.8455686701969342787869758198351951379156813281202e-1000'