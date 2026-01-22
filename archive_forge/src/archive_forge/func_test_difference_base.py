import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_difference_base(idx, sort):
    second = idx[4:]
    answer = idx[:4]
    result = idx.difference(second, sort=sort)
    if sort is None:
        answer = answer.sort_values()
    assert result.equals(answer)
    tm.assert_index_equal(result, answer)
    cases = [klass(second.values) for klass in [np.array, Series, list]]
    for case in cases:
        result = idx.difference(case, sort=sort)
        tm.assert_index_equal(result, answer)
    msg = 'other must be a MultiIndex or a list of tuples'
    with pytest.raises(TypeError, match=msg):
        idx.difference([1, 2, 3], sort=sort)