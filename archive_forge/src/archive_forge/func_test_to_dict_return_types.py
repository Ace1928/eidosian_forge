import collections
import numpy as np
import pytest
from pandas import Series
import pandas._testing as tm
@pytest.mark.parametrize('input', ({'a': np.int64(64), 'b': 10}, {'a': np.int64(64), 'b': 10, 'c': 'ABC'}, {'a': np.uint64(64), 'b': 10, 'c': 'ABC'}))
def test_to_dict_return_types(self, input):
    d = Series(input).to_dict()
    assert isinstance(d['a'], int)
    assert isinstance(d['b'], int)