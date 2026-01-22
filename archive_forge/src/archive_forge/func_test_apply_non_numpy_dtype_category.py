from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_non_numpy_dtype_category():
    df = DataFrame({'dt': ['a', 'b', 'c', 'a']}, dtype='category')
    result = df.apply(lambda x: x)
    tm.assert_frame_equal(result, df)