from symengine.test_utilities import raises
from symengine import Integer, I, S, Symbol, pi, Rational
from symengine.lib.symengine_wrapper import (perfect_power, is_square, integer_nthroot)
def test_is_nonpositive():
    assert not Rational(1, 2).is_nonpositive
    assert Rational(-2, 3).is_nonpositive
    assert Symbol('x').is_nonpositive is None