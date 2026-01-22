from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_on_empty_dataframe(engine):
    df = DataFrame({'a': [1, 2], 'b': [3, 0]})
    result = df.head(0).apply(lambda x: max(x['a'], x['b']), axis=1, engine=engine)
    expected = Series([], dtype=np.float64)
    tm.assert_series_equal(result, expected)