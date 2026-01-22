from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_apply_with_byte_string():
    df = DataFrame(np.array([b'abcd', b'efgh']), columns=['col'])
    expected = DataFrame(np.array([b'abcd', b'efgh']), columns=['col'], dtype=object)
    result = df.apply(lambda x: x.astype('object'))
    tm.assert_frame_equal(result, expected)