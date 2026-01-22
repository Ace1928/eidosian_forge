from datetime import (
import operator
import numpy as np
import pytest
from pandas import Timestamp
import pandas._testing as tm
def test_compare_non_nano_dt64(self):
    dt = np.datetime64('1066-10-14')
    ts = Timestamp(dt)
    assert dt == ts