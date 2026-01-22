import pytest
import pickle
from numpy.testing import assert_equal
from scipy._lib._bunch import _make_tuple_bunch
@pytest.mark.parametrize('args', [('123', ['a'], ['b']), ('Foo', ['-3'], ['x']), ('Foo', ['a'], ['+-*/'])])
def test_identifiers_not_allowed(self, args):
    with pytest.raises(ValueError, match='identifiers'):
        _make_tuple_bunch(*args)