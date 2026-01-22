from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.typing import SeriesGroupBy
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('observed', [False, None])
def test_dataframe_groupby_on_2_categoricals_when_observed_is_false(reduction_func, observed):
    if reduction_func == 'ngroup':
        pytest.skip('ngroup does not return the Categories on the index')
    df = DataFrame({'cat_1': Categorical(list('AABB'), categories=list('ABC')), 'cat_2': Categorical(list('1111'), categories=list('12')), 'value': [0.1, 0.1, 0.1, 0.1]})
    unobserved_cats = [('A', '2'), ('B', '2'), ('C', '1'), ('C', '2')]
    df_grp = df.groupby(['cat_1', 'cat_2'], observed=observed)
    args = get_groupby_method_args(reduction_func, df)
    if not observed and reduction_func in ['idxmin', 'idxmax']:
        with pytest.raises(ValueError, match='empty group due to unobserved categories'):
            getattr(df_grp, reduction_func)(*args)
        return
    res = getattr(df_grp, reduction_func)(*args)
    expected = _results_for_groupbys_with_missing_categories[reduction_func]
    if expected is np.nan:
        assert res.loc[unobserved_cats].isnull().all().all()
    else:
        assert (res.loc[unobserved_cats] == expected).all().all()