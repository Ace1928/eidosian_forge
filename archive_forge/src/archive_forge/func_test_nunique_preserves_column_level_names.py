import datetime as dt
from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_nunique_preserves_column_level_names():
    test = DataFrame([1, 2, 2], columns=pd.Index(['A'], name='level_0'))
    result = test.groupby([0, 0, 0]).nunique()
    expected = DataFrame([2], index=np.array([0]), columns=test.columns)
    tm.assert_frame_equal(result, expected)