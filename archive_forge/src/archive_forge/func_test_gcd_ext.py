from symengine.test_utilities import raises
from symengine.lib.symengine_wrapper import (isprime, nextprime, gcd,
def test_gcd_ext():
    q, r, p = gcd_ext(6, 9)
    assert p == q * 6 + r * 9
    q, r, p = gcd_ext(-15, 10)
    assert p == q * -15 + r * 10
    q, r, p = gcd_ext(2, 3)
    assert p == q * 2 + r * 3
    assert p == 1
    q, r, p = gcd_ext(10, 12)
    assert p == q * 10 + r * 12
    assert p == 2
    q, r, p = gcd_ext(100, 2004)
    assert p == q * 100 + r * 2004
    assert p == 4