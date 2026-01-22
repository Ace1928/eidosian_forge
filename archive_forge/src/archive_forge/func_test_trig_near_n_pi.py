from mpmath import *
from mpmath.libmp import *
def test_trig_near_n_pi():
    mp.dps = 15
    a = [n * pi for n in [1, 2, 6, 11, 100, 1001, 10000, 100001]]
    mp.dps = 135
    a.append(10 ** 100 * pi)
    mp.dps = 15
    assert sin(a[0]) == mpf('1.2246467991473531772e-16')
    assert sin(a[1]) == mpf('-2.4492935982947063545e-16')
    assert sin(a[2]) == mpf('-7.3478807948841190634e-16')
    assert sin(a[3]) == mpf('4.8998251578625894243e-15')
    assert sin(a[4]) == mpf('1.9643867237284719452e-15')
    assert sin(a[5]) == mpf('-8.8632615209684813458e-15')
    assert sin(a[6]) == mpf('-4.8568235395684898392e-13')
    assert sin(a[7]) == mpf('3.9087342299491231029e-11')
    assert sin(a[8]) == mpf('-1.369235466754566993528e-36')
    r = round_nearest
    assert cos(a[0], rounding=r) == -1
    assert cos(a[1], rounding=r) == 1
    assert cos(a[2], rounding=r) == 1
    assert cos(a[3], rounding=r) == -1
    assert cos(a[4], rounding=r) == 1
    assert cos(a[5], rounding=r) == -1
    assert cos(a[6], rounding=r) == 1
    assert cos(a[7], rounding=r) == -1
    assert cos(a[8], rounding=r) == 1
    r = round_up
    assert cos(a[0], rounding=r) == -1
    assert cos(a[1], rounding=r) == 1
    assert cos(a[2], rounding=r) == 1
    assert cos(a[3], rounding=r) == -1
    assert cos(a[4], rounding=r) == 1
    assert cos(a[5], rounding=r) == -1
    assert cos(a[6], rounding=r) == 1
    assert cos(a[7], rounding=r) == -1
    assert cos(a[8], rounding=r) == 1
    r = round_down
    assert cos(a[0], rounding=r) > -1
    assert cos(a[1], rounding=r) < 1
    assert cos(a[2], rounding=r) < 1
    assert cos(a[3], rounding=r) > -1
    assert cos(a[4], rounding=r) < 1
    assert cos(a[5], rounding=r) > -1
    assert cos(a[6], rounding=r) < 1
    assert cos(a[7], rounding=r) > -1
    assert cos(a[8], rounding=r) < 1
    r = round_floor
    assert cos(a[0], rounding=r) == -1
    assert cos(a[1], rounding=r) < 1
    assert cos(a[2], rounding=r) < 1
    assert cos(a[3], rounding=r) == -1
    assert cos(a[4], rounding=r) < 1
    assert cos(a[5], rounding=r) == -1
    assert cos(a[6], rounding=r) < 1
    assert cos(a[7], rounding=r) == -1
    assert cos(a[8], rounding=r) < 1
    r = round_ceiling
    assert cos(a[0], rounding=r) > -1
    assert cos(a[1], rounding=r) == 1
    assert cos(a[2], rounding=r) == 1
    assert cos(a[3], rounding=r) > -1
    assert cos(a[4], rounding=r) == 1
    assert cos(a[5], rounding=r) > -1
    assert cos(a[6], rounding=r) == 1
    assert cos(a[7], rounding=r) > -1
    assert cos(a[8], rounding=r) == 1
    mp.dps = 15