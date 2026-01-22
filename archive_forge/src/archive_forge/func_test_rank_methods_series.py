from itertools import chain
import operator
import numpy as np
import pytest
from pandas._libs.algos import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
@pytest.mark.parametrize('method', ['average', 'min', 'max', 'first', 'dense'])
@pytest.mark.parametrize('op, value', [[operator.add, 0], [operator.add, 1000000.0], [operator.mul, 1e-06]])
def test_rank_methods_series(self, method, op, value):
    sp_stats = pytest.importorskip('scipy.stats')
    xs = np.random.default_rng(2).standard_normal(9)
    xs = np.concatenate([xs[i:] for i in range(0, 9, 2)])
    np.random.default_rng(2).shuffle(xs)
    index = [chr(ord('a') + i) for i in range(len(xs))]
    vals = op(xs, value)
    ts = Series(vals, index=index)
    result = ts.rank(method=method)
    sprank = sp_stats.rankdata(vals, method if method != 'first' else 'ordinal')
    expected = Series(sprank, index=index).astype('float64')
    tm.assert_series_equal(result, expected)