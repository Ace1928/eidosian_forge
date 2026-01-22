import numpy as np
import pytest
from pandas.errors import NumbaUtilError
from pandas import (
import pandas._testing as tm
@pytest.mark.single_cpu
@pytest.mark.parametrize('data,agg_kwargs', [(Series([1.0, 2.0, 3.0, 4.0, 5.0]), {'func': ['min', 'max']}), (Series([1.0, 2.0, 3.0, 4.0, 5.0]), {'func': 'min'}), (DataFrame({1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]), {'func': ['min', 'max']}), (DataFrame({1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]), {'func': 'min'}), (DataFrame({1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]), {'func': {1: ['min', 'max'], 2: 'sum'}}), (DataFrame({1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]), {'min_col': NamedAgg(column=1, aggfunc='min')})])
def test_multifunc_numba_kwarg_propagation(data, agg_kwargs):
    pytest.importorskip('numba')
    labels = ['a', 'a', 'b', 'b', 'a']
    grouped = data.groupby(labels)
    result = grouped.agg(**agg_kwargs, engine='numba', engine_kwargs={'parallel': True})
    expected = grouped.agg(**agg_kwargs, engine='numba')
    if isinstance(expected, DataFrame):
        tm.assert_frame_equal(result, expected)
    else:
        tm.assert_series_equal(result, expected)