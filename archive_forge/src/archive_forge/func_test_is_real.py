from symengine.test_utilities import raises
from symengine import Integer, I, S, Symbol, pi, Rational
from symengine.lib.symengine_wrapper import (perfect_power, is_square, integer_nthroot)
def test_is_real():
    assert Rational(1, 2).is_real
    assert Symbol('x').is_real is None