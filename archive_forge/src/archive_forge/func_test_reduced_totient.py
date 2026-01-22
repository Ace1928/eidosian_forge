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
def test_reduced_totient():
    assert [reduced_totient(k) for k in range(1, 16)] == [1, 1, 2, 2, 4, 2, 6, 2, 6, 4, 10, 2, 12, 6, 4]
    assert reduced_totient(5005) == 60
    assert reduced_totient(5006) == 2502
    assert reduced_totient(5009) == 5008
    assert reduced_totient(2 ** 100) == 2 ** 98
    m = Symbol('m', integer=True)
    assert reduced_totient(m)
    assert reduced_totient(m).subs(m, 2 ** 3 * 3 ** 10) == 3 ** 10 - 3 ** 9
    assert summation(reduced_totient(m), (m, 1, 16)) == 68
    n = Symbol('n', integer=True, positive=True)
    assert reduced_totient(n).is_integer