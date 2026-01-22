import pytest
import pickle
from numpy.testing import assert_equal
from scipy._lib._bunch import _make_tuple_bunch
@pytest.mark.parametrize('args', [('Foo', ['def'], ['x']), ('Foo', ['a'], ['or']), ('and', ['a'], ['x'])])
def test_keyword_not_allowed_in_fields(self, args):
    with pytest.raises(ValueError, match='keyword'):
        _make_tuple_bunch(*args)