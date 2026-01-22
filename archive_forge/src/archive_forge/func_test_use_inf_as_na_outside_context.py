import collections
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('values, expected', [([1, 2, 3], np.array([False, False, False])), ([1, 2, np.nan], np.array([False, False, True])), ([1, 2, np.inf], np.array([False, False, True])), ([1, 2, pd.NA], np.array([False, False, True]))])
def test_use_inf_as_na_outside_context(self, values, expected):
    cat = Categorical(values)
    msg = 'use_inf_as_na option is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pd.option_context('mode.use_inf_as_na', True):
            result = isna(cat)
            tm.assert_numpy_array_equal(result, expected)
            result = isna(Series(cat))
            expected = Series(expected)
            tm.assert_series_equal(result, expected)
            result = isna(DataFrame(cat))
            expected = DataFrame(expected)
            tm.assert_frame_equal(result, expected)