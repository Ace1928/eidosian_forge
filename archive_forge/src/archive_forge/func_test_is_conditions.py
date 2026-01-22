from symengine.test_utilities import raises
from symengine import Integer, I, S, Symbol, pi, Rational
from symengine.lib.symengine_wrapper import (perfect_power, is_square, integer_nthroot)
def test_is_conditions():
    i = Integer(-123)
    assert not i.is_zero
    assert not i.is_positive
    assert i.is_negative
    assert i.is_nonzero
    assert i.is_nonpositive
    assert not i.is_nonnegative
    assert not i.is_complex
    i = Integer(123)
    assert not i.is_zero
    assert i.is_positive
    assert not i.is_negative
    assert i.is_nonzero
    assert not i.is_nonpositive
    assert i.is_nonnegative
    assert not i.is_complex
    i = Integer(0)
    assert i.is_zero
    assert not i.is_positive
    assert not i.is_negative
    assert not i.is_nonzero
    assert i.is_nonpositive
    assert i.is_nonnegative
    assert not i.is_complex
    i = Integer(1) + I
    assert not i.is_zero
    assert not i.is_positive
    assert not i.is_negative
    assert not i.is_nonzero
    assert not i.is_nonpositive
    assert not i.is_nonnegative
    assert i.is_complex
    assert pi.is_number