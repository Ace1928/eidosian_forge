from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
def test_map_box_timestamps():
    ser = Series(date_range('1/1/2000', periods=10))

    def func(x):
        return (x.hour, x.day, x.month)
    DataFrame(ser).map(func)