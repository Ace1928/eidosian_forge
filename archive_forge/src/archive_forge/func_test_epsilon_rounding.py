from mpmath.libmp import *
from mpmath import mpf, mp
from random import randint, choice, seed
def test_epsilon_rounding():
    a = fb('0.101' + '0' * 200 + '1')
    b = fb('1.10101')
    c = mpf_mul(a, b, 250, round_floor)
    assert mpf_div(c, b, bitcount(a[1]), round_floor) == a
    assert mpf_div(c, b, 2, round_down) == fb('0.10')
    assert mpf_div(c, b, 3, round_down) == fb('0.101')
    assert mpf_div(c, b, 2, round_up) == fb('0.11')
    assert mpf_div(c, b, 3, round_up) == fb('0.110')
    assert mpf_div(c, b, 2, round_floor) == fb('0.10')
    assert mpf_div(c, b, 3, round_floor) == fb('0.101')
    assert mpf_div(c, b, 2, round_ceiling) == fb('0.11')
    assert mpf_div(c, b, 3, round_ceiling) == fb('0.110')
    a = fb('-0.101' + '0' * 200 + '1')
    b = fb('1.10101')
    c = mpf_mul(a, b, 250, round_floor)
    assert mpf_div(c, b, bitcount(a[1]), round_floor) == a
    assert mpf_div(c, b, 2, round_down) == fb('-0.10')
    assert mpf_div(c, b, 3, round_up) == fb('-0.110')
    assert mpf_div(c, b, 2, round_floor) == fb('-0.11')
    assert mpf_div(c, b, 3, round_floor) == fb('-0.110')
    assert mpf_div(c, b, 2, round_ceiling) == fb('-0.10')
    assert mpf_div(c, b, 3, round_ceiling) == fb('-0.101')