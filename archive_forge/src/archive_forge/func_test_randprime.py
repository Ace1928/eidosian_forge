from sympy.core.numbers import (I, Rational, nan, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.ntheory.generate import (sieve, Sieve)
from sympy.series.limits import limit
from sympy.ntheory import isprime, totient, mobius, randprime, nextprime, prevprime, \
from sympy.ntheory.generate import cycle_length
from sympy.ntheory.primetest import mr
from sympy.testing.pytest import raises
def test_randprime():
    assert randprime(10, 1) is None
    assert randprime(3, -3) is None
    assert randprime(2, 3) == 2
    assert randprime(1, 3) == 2
    assert randprime(3, 5) == 3
    raises(ValueError, lambda: randprime(-12, -2))
    raises(ValueError, lambda: randprime(-10, 0))
    raises(ValueError, lambda: randprime(20, 22))
    raises(ValueError, lambda: randprime(0, 2))
    raises(ValueError, lambda: randprime(1, 2))
    for a in [100, 300, 500, 250000]:
        for b in [100, 300, 500, 250000]:
            p = randprime(a, a + b)
            assert a <= p < a + b and isprime(p)