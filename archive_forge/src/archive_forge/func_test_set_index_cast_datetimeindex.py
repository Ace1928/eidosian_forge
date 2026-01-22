from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_set_index_cast_datetimeindex(self):
    df = DataFrame({'A': [datetime(2000, 1, 1) + timedelta(i) for i in range(1000)], 'B': np.random.default_rng(2).standard_normal(1000)})
    idf = df.set_index('A')
    assert isinstance(idf.index, DatetimeIndex)