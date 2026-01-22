from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_axis1_with_ea():
    expected = DataFrame({'A': [Timestamp('2013-01-01', tz='UTC')]})
    result = expected.apply(lambda x: x, axis=1)
    tm.assert_frame_equal(result, expected)