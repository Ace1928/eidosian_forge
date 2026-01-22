import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ind', [['a', 'b', 'c', 'd'], pd.period_range(start='2019-01-01', periods=4), pd.interval_range(start=0, end=4)])
def test_interp_non_timedelta_index(self, interp_methods_ind, ind):
    df = pd.DataFrame([0, 1, np.nan, 3], index=ind)
    method, kwargs = interp_methods_ind
    if method == 'pchip':
        pytest.importorskip('scipy')
    if method == 'linear':
        result = df[0].interpolate(**kwargs)
        expected = Series([0.0, 1.0, 2.0, 3.0], name=0, index=ind)
        tm.assert_series_equal(result, expected)
    else:
        expected_error = f'Index column must be numeric or datetime type when using {method} method other than linear. Try setting a numeric or datetime index column before interpolating.'
        with pytest.raises(ValueError, match=expected_error):
            df[0].interpolate(method=method, **kwargs)