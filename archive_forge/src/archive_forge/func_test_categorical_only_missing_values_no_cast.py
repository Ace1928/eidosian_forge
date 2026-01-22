import collections
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('na_value, dtype', [(pd.NaT, 'datetime64[ns]'), (None, 'float64'), (np.nan, 'float64'), (pd.NA, 'float64')])
def test_categorical_only_missing_values_no_cast(self, na_value, dtype):
    result = Categorical([na_value, na_value])
    tm.assert_index_equal(result.categories, Index([], dtype=dtype))