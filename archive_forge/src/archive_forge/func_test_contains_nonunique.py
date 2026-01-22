from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
@pytest.mark.parametrize('vals', [[0, 1, 0], [0, 0, -1], [0, -1, -1], ['2015', '2015', '2016'], ['2015', '2015', '2014']])
def test_contains_nonunique(self, vals):
    idx = DatetimeIndex(vals)
    assert idx[0] in idx