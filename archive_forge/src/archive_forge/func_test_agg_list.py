from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('test_series', [True, False])
@pytest.mark.parametrize('keys', [['a1'], ['a1', 'a2']])
def test_agg_list(request, as_index, observed, reduction_func, test_series, keys):
    if test_series and reduction_func == 'corrwith':
        assert not hasattr(SeriesGroupBy, 'corrwith')
        pytest.skip('corrwith not implemented for SeriesGroupBy')
    elif reduction_func == 'corrwith':
        msg = 'GH#32293: attempts to call SeriesGroupBy.corrwith'
        request.applymarker(pytest.mark.xfail(reason=msg))
    elif reduction_func == 'nunique' and (not test_series) and (len(keys) != 1) and (not observed) and (not as_index):
        msg = 'GH#52848 - raises a ValueError'
        request.applymarker(pytest.mark.xfail(reason=msg))
    df = DataFrame({'a1': [0, 0, 1], 'a2': [2, 3, 3], 'b': [4, 5, 6]})
    df = df.astype({'a1': 'category', 'a2': 'category'})
    if 'a2' not in keys:
        df = df.drop(columns='a2')
    gb = df.groupby(by=keys, as_index=as_index, observed=observed)
    if test_series:
        gb = gb['b']
    args = get_groupby_method_args(reduction_func, df)
    if not observed and reduction_func in ['idxmin', 'idxmax'] and (keys == ['a1', 'a2']):
        with pytest.raises(ValueError, match='empty group due to unobserved categories'):
            gb.agg([reduction_func], *args)
        return
    result = gb.agg([reduction_func], *args)
    expected = getattr(gb, reduction_func)(*args)
    if as_index and (test_series or reduction_func == 'size'):
        expected = expected.to_frame(reduction_func)
    if not test_series:
        expected.columns = MultiIndex.from_tuples([(ind, '') for ind in expected.columns[:-1]] + [('b', reduction_func)])
    elif not as_index:
        expected.columns = keys + [reduction_func]
    tm.assert_equal(result, expected)