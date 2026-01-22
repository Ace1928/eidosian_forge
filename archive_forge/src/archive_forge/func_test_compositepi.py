from sympy.core.numbers import (I, Rational, nan, zoo)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.ntheory.generate import (sieve, Sieve)
from sympy.series.limits import limit
from sympy.ntheory import isprime, totient, mobius, randprime, nextprime, prevprime, \
from sympy.ntheory.generate import cycle_length
from sympy.ntheory.primetest import mr
from sympy.testing.pytest import raises
def test_compositepi():
    assert compositepi(1) == 0
    assert compositepi(2) == 0
    assert compositepi(5) == 1
    assert compositepi(11) == 5
    assert compositepi(57) == 40
    assert compositepi(296) == 233
    assert compositepi(559) == 456
    assert compositepi(3000) == 2569
    assert compositepi(4096) == 3531
    assert compositepi(9096) == 7967
    assert compositepi(25023) == 22259
    assert compositepi(10 ** 8) == 94238544
    assert compositepi(253425253) == 239568856
    assert compositepi(8769575643) == 8368111320
    sieve.extend(3000)
    assert compositepi(2321) == 1976