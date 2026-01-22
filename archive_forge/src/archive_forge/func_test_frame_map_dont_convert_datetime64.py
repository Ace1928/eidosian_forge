from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_frame_map_dont_convert_datetime64():
    df = DataFrame({'x1': [datetime(1996, 1, 1)]})
    df = df.map(lambda x: x + BDay())
    df = df.map(lambda x: x + BDay())
    result = df.x1.dtype
    assert result == 'M8[ns]'