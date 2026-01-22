import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_cumprod_smoke(self, datetime_frame):
    datetime_frame.iloc[5:10, 0] = np.nan
    datetime_frame.iloc[10:15, 1] = np.nan
    datetime_frame.iloc[15:, 2] = np.nan
    df = datetime_frame.fillna(0).astype(int)
    df.cumprod(0)
    df.cumprod(1)
    df = datetime_frame.fillna(0).astype(np.int32)
    df.cumprod(0)
    df.cumprod(1)