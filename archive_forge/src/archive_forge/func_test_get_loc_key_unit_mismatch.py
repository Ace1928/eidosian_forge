from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
def test_get_loc_key_unit_mismatch(self):
    idx = date_range('2000-01-01', periods=3)
    key = idx[1].as_unit('ms')
    loc = idx.get_loc(key)
    assert loc == 1
    assert key in idx