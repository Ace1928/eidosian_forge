import mpmath
import random
import pytest
from mpmath import *
def test_jsn():
    """
    Test some special cases of the sn(z, q) function.
    """
    mp.dps = 100
    result = jsn(zero, zero)
    assert result == zero
    for i in range(10):
        qstring = str(random.random())
        q = mpf(qstring)
        equality = jsn(zero, q)
        assert equality.ae(0)
    mp.dps = 25
    arg = one / 10
    res = mpf('0.09983341664682815230681420')
    m = ldexp(one, -100)
    result = jsn(arg, m)
    assert result.ae(res)
    res = mpf('0.09981686718599080096451168')
    result = jsn(arg, arg)
    assert result.ae(res)
    mp.dps = 15