from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_return_list():
    df = DataFrame({'a': [1, 2], 'b': [2, 3]})
    result = df.apply(lambda x: [x.values])
    expected = DataFrame({'a': [[1, 2]], 'b': [[2, 3]]})
    tm.assert_frame_equal(result, expected)