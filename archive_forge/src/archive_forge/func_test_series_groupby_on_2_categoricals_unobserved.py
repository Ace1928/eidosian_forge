from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_series_groupby_on_2_categoricals_unobserved(reduction_func, observed):
    if reduction_func == 'ngroup':
        pytest.skip('ngroup is not truly a reduction')
    df = DataFrame({'cat_1': Categorical(list('AABB'), categories=list('ABCD')), 'cat_2': Categorical(list('AB') * 2, categories=list('ABCD')), 'value': [0.1] * 4})
    args = get_groupby_method_args(reduction_func, df)
    expected_length = 4 if observed else 16
    series_groupby = df.groupby(['cat_1', 'cat_2'], observed=observed)['value']
    if reduction_func == 'corrwith':
        assert not hasattr(series_groupby, reduction_func)
        return
    agg = getattr(series_groupby, reduction_func)
    if not observed and reduction_func in ['idxmin', 'idxmax']:
        with pytest.raises(ValueError, match='empty group due to unobserved categories'):
            agg(*args)
        return
    result = agg(*args)
    assert len(result) == expected_length