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
def test_visual_factorint():
    assert factorint(1, visual=1) == 1
    forty2 = factorint(42, visual=True)
    assert type(forty2) == Mul
    assert str(forty2) == '2**1*3**1*7**1'
    assert factorint(1, visual=True) is S.One
    no = {'evaluate': False}
    assert factorint(42 ** 2, visual=True) == Mul(Pow(2, 2, **no), Pow(3, 2, **no), Pow(7, 2, **no), **no)
    assert -1 in factorint(-42, visual=True).args