from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_from_index_series_datetimetz(self):
    idx = date_range('2015-01-01 10:00', freq='D', periods=3, tz='US/Eastern')
    idx = idx._with_freq(None)
    result = Categorical(idx)
    tm.assert_index_equal(result.categories, idx)
    result = Categorical(Series(idx))
    tm.assert_index_equal(result.categories, idx)