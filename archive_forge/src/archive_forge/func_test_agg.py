import numpy as np
import pytest
from pandas.errors import (
from pandas import (
import pandas._testing as tm
def test_agg(step):
    df = DataFrame({'A': range(5), 'B': range(0, 10, 2)})
    r = df.rolling(window=3, step=step)
    a_mean = r['A'].mean()
    a_std = r['A'].std()
    a_sum = r['A'].sum()
    b_mean = r['B'].mean()
    b_std = r['B'].std()
    with tm.assert_produces_warning(FutureWarning, match='using Rolling.[mean|std]'):
        result = r.aggregate([np.mean, np.std])
    expected = concat([a_mean, a_std, b_mean, b_std], axis=1)
    expected.columns = MultiIndex.from_product([['A', 'B'], ['mean', 'std']])
    tm.assert_frame_equal(result, expected)
    with tm.assert_produces_warning(FutureWarning, match='using Rolling.[mean|std]'):
        result = r.aggregate({'A': np.mean, 'B': np.std})
    expected = concat([a_mean, b_std], axis=1)
    tm.assert_frame_equal(result, expected, check_like=True)
    result = r.aggregate({'A': ['mean', 'std']})
    expected = concat([a_mean, a_std], axis=1)
    expected.columns = MultiIndex.from_tuples([('A', 'mean'), ('A', 'std')])
    tm.assert_frame_equal(result, expected)
    result = r['A'].aggregate(['mean', 'sum'])
    expected = concat([a_mean, a_sum], axis=1)
    expected.columns = ['mean', 'sum']
    tm.assert_frame_equal(result, expected)
    msg = 'nested renamer is not supported'
    with pytest.raises(SpecificationError, match=msg):
        r.aggregate({'A': {'mean': 'mean', 'sum': 'sum'}})
    with pytest.raises(SpecificationError, match=msg):
        r.aggregate({'A': {'mean': 'mean', 'sum': 'sum'}, 'B': {'mean2': 'mean', 'sum2': 'sum'}})
    result = r.aggregate({'A': ['mean', 'std'], 'B': ['mean', 'std']})
    expected = concat([a_mean, a_std, b_mean, b_std], axis=1)
    exp_cols = [('A', 'mean'), ('A', 'std'), ('B', 'mean'), ('B', 'std')]
    expected.columns = MultiIndex.from_tuples(exp_cols)
    tm.assert_frame_equal(result, expected, check_like=True)