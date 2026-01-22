from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
def test_series_groupby_on_2_categoricals_unobserved_zeroes_or_nans(reduction_func, request):
    if reduction_func == 'ngroup':
        pytest.skip('ngroup is not truly a reduction')
    if reduction_func == 'corrwith':
        mark = pytest.mark.xfail(reason='TODO: implemented SeriesGroupBy.corrwith. See GH 32293')
        request.applymarker(mark)
    df = DataFrame({'cat_1': Categorical(list('AABB'), categories=list('ABC')), 'cat_2': Categorical(list('AB') * 2, categories=list('ABC')), 'value': [0.1] * 4})
    unobserved = [tuple('AC'), tuple('BC'), tuple('CA'), tuple('CB'), tuple('CC')]
    args = get_groupby_method_args(reduction_func, df)
    series_groupby = df.groupby(['cat_1', 'cat_2'], observed=False)['value']
    agg = getattr(series_groupby, reduction_func)
    if reduction_func in ['idxmin', 'idxmax']:
        with pytest.raises(ValueError, match='empty group due to unobserved categories'):
            agg(*args)
        return
    result = agg(*args)
    zero_or_nan = _results_for_groupbys_with_missing_categories[reduction_func]
    for idx in unobserved:
        val = result.loc[idx]
        assert pd.isna(zero_or_nan) and pd.isna(val) or val == zero_or_nan
    if zero_or_nan == 0 and reduction_func != 'sum':
        assert np.issubdtype(result.dtype, np.integer)