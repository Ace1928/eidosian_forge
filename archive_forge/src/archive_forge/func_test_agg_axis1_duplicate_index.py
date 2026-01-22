from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
@pytest.mark.parametrize('data, dtype', [(1, None), (1, CategoricalDtype([1])), (Timestamp('2013-01-01', tz='UTC'), None)])
def test_agg_axis1_duplicate_index(data, dtype):
    expected = DataFrame([[data], [data]], index=['a', 'a'], dtype=dtype)
    result = expected.agg(lambda x: x, axis=1)
    tm.assert_frame_equal(result, expected)