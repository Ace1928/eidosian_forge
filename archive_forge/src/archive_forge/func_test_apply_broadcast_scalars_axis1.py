from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_broadcast_scalars_axis1(float_frame):
    result = float_frame.apply(np.mean, axis=1, result_type='broadcast')
    m = float_frame.mean(axis=1)
    expected = DataFrame({c: m for c in float_frame.columns})
    tm.assert_frame_equal(result, expected)