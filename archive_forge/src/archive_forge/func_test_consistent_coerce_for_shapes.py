from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('lst', [[1, 2, 3], [1, 2]])
def test_consistent_coerce_for_shapes(lst):
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 3)), columns=['A', 'B', 'C'])
    result = df.apply(lambda x: lst, axis=1)
    expected = Series([lst for t in df.itertuples()])
    tm.assert_series_equal(result, expected)