import pytest
import pickle
from numpy.testing import assert_equal
from scipy._lib._bunch import _make_tuple_bunch
def test_extra_field_names_is_optional(self):
    Square = _make_tuple_bunch('Square', ['width', 'height'])
    sq = Square(width=1, height=2)
    assert_equal(sq.width, 1)
    assert_equal(sq.height, 2)
    s = repr(sq)
    assert_equal(s, 'Square(width=1, height=2)')