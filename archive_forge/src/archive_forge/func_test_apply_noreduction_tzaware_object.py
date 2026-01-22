from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_noreduction_tzaware_object():
    expected = DataFrame({'foo': [Timestamp('2020', tz='UTC')]}, dtype='datetime64[ns, UTC]')
    result = expected.apply(lambda x: x)
    tm.assert_frame_equal(result, expected)
    result = expected.apply(lambda x: x.copy())
    tm.assert_frame_equal(result, expected)