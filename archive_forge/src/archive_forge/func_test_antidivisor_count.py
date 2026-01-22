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
def test_antidivisor_count():
    assert antidivisor_count(0) == 0
    assert antidivisor_count(-1) == 0
    assert antidivisor_count(-4) == 1
    assert antidivisor_count(20) == 3
    assert antidivisor_count(25) == 5
    assert antidivisor_count(38) == 7
    assert antidivisor_count(180) == 6
    assert antidivisor_count(2 * 3 * 5) == 3