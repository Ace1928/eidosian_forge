from sympy.concrete.summations import summation
from sympy.core.containers import Dict
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import factorial as fac
from sympy.core.evalf import bitcount
from sympy.core.numbers import Integer, Rational
from sympy.ntheory import (totient,
from sympy.ntheory.factor_ import (smoothness, smoothness_p, proper_divisors,
from sympy.testing.pytest import raises, slow
from sympy.utilities.iterables import capture
def test_totient():
    assert [totient(k) for k in range(1, 12)] == [1, 1, 2, 2, 4, 2, 6, 4, 6, 4, 10]
    assert totient(5005) == 2880
    assert totient(5006) == 2502
    assert totient(5009) == 5008
    assert totient(2 ** 100) == 2 ** 99
    raises(ValueError, lambda: totient(30.1))
    raises(ValueError, lambda: totient(20.001))
    m = Symbol('m', integer=True)
    assert totient(m)
    assert totient(m).subs(m, 3 ** 10) == 3 ** 10 - 3 ** 9
    assert summation(totient(m), (m, 1, 11)) == 42
    n = Symbol('n', integer=True, positive=True)
    assert totient(n).is_integer
    x = Symbol('x', integer=False)
    raises(ValueError, lambda: totient(x))
    y = Symbol('y', positive=False)
    raises(ValueError, lambda: totient(y))
    z = Symbol('z', positive=True, integer=True)
    raises(ValueError, lambda: totient(2 ** (-z)))