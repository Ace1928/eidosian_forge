import time
from mpmath import *
def test_log_hp():
    mp.dps = 2000
    a = mpf(10) ** 15000 / 3
    r = log(a)
    res = last_digits(r)
    assert res == '43804441768333470331'
    r = log(mpf(3) / 2)
    res = last_digits(r)
    assert res == '53749808140753263288'
    mp.dps = 10000
    r = log(2)
    res = last_digits(r)
    assert res == '13401856601359655561'
    r = log(mpf(10) ** 10 / 3)
    res = last_digits(r)
    assert res == '54020631943060007154', res
    r = log(mpf(10) ** 100 / 3)
    res = last_digits(r)
    assert res == '36539088351652334666', res
    mp.dps += 10
    a = 1 - mpf(1) / 10 ** 10
    mp.dps -= 10
    r = log(a)
    res = last_digits(r)
    assert res == '37216724048322957404', res
    mp.dps = 10000
    mp.dps += 100
    a = 1 + mpf(1) / 10 ** 100
    mp.dps -= 100
    r = log(a)
    res = last_digits(+r)
    assert res == '39947338773774122415', res
    mp.dps = 15