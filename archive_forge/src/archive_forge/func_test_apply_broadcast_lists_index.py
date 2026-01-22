from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_broadcast_lists_index(float_frame):
    result = float_frame.apply(lambda x: list(range(len(float_frame.index))), result_type='broadcast')
    m = list(range(len(float_frame.index)))
    expected = DataFrame({c: m for c in float_frame.columns}, dtype='float64', index=float_frame.index)
    tm.assert_frame_equal(result, expected)