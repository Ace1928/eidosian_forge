from mpmath import *
from mpmath.libmp import round_up, from_float, mpf_zeta_int
def test_zeta_near_1():
    mp.dps = 15
    s1 = fadd(1, '1e-10', exact=True)
    s2 = fadd(1, '-1e-10', exact=True)
    s3 = fadd(1, '1e-10j', exact=True)
    assert zeta(s1).ae(10000000000.577215)
    assert zeta(s2).ae(-9999999999.422785)
    z = zeta(s3)
    assert z.real.ae(0.5772156649015329)
    assert z.imag.ae(-10000000000.0)
    mp.dps = 30
    s1 = fadd(1, '1e-50', exact=True)
    s2 = fadd(1, '-1e-50', exact=True)
    s3 = fadd(1, '1e-50j', exact=True)
    assert zeta(s1).ae('1e50')
    assert zeta(s2).ae('-1e50')
    z = zeta(s3)
    assert z.real.ae('0.57721566490153286060651209008240243104215933593992')
    assert z.imag.ae('-1e50')