from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
def test_get_indexer_pad_requires_monotonicity(self):
    rng = date_range('1/1/2000', '3/1/2000', freq='B')
    rng2 = rng[[1, 0, 2]]
    msg = 'index must be monotonic increasing or decreasing'
    with pytest.raises(ValueError, match=msg):
        rng2.get_indexer(rng, method='pad')