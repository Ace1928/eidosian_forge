import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under10p1
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
@pytest.mark.parametrize('method', ['head', 'tail'])
def test_categorical_head_tail(method, observed, sort, as_index):
    values = np.random.default_rng(2).choice([1, 2, None], 30)
    df = pd.DataFrame({'x': pd.Categorical(values, categories=[1, 2, 3]), 'y': range(len(values))})
    gb = df.groupby('x', dropna=False, observed=observed, sort=sort, as_index=as_index)
    result = getattr(gb, method)()
    if method == 'tail':
        values = values[::-1]
    mask = (values == 1) & ((values == 1).cumsum() <= 5) | (values == 2) & ((values == 2).cumsum() <= 5) | (values == None) & ((values == None).cumsum() <= 5)
    if method == 'tail':
        mask = mask[::-1]
    expected = df[mask]
    tm.assert_frame_equal(result, expected)