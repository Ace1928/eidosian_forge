from mpmath import *
from mpmath.libmp import *
def test_trig_misc_hard():
    mp.prec = 53
    x = ldexp(6381956970095103, 797)
    assert cos(x) == mpf('-4.6871659242546277e-19')
    assert sin(x) == 1
    mp.prec = 150
    a = mpf(10 ** 50)
    mp.prec = 53
    assert sin(a).ae(-0.7896724934293101)
    assert cos(a).ae(-0.6135286082336635)
    assert sin(1e-100) == 1e-100
    assert sin(1e-06).ae(9.999999999998333e-07, rel_eps=2e-15, abs_eps=0)
    assert sin(1e-06j).ae(1.0000000000001666e-06j, rel_eps=2e-15, abs_eps=0)
    assert sin(-1e-06j).ae(-1.0000000000001666e-06j, rel_eps=2e-15, abs_eps=0)
    assert cos(1e-100) == 1
    assert cos(1e-06).ae(0.9999999999995)
    assert cos(-1e-06j).ae(1.0000000000005)
    assert tan(1e-100) == 1e-100
    assert tan(1e-06).ae(1.0000000000003335e-06, rel_eps=2e-15, abs_eps=0)
    assert tan(1e-06j).ae(9.999999999996664e-07j, rel_eps=2e-15, abs_eps=0)
    assert tan(-1e-06j).ae(-9.999999999996664e-07j, rel_eps=2e-15, abs_eps=0)