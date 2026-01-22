from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_slice_locs_with_type_mismatch(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
    stacked = df.stack(future_stack=True)
    idx = stacked.index
    with pytest.raises(TypeError, match='^Level type mismatch'):
        idx.slice_locs((1, 3))
    with pytest.raises(TypeError, match='^Level type mismatch'):
        idx.slice_locs(df.index[5] + timedelta(seconds=30), (5, 2))
    df = DataFrame(np.ones((5, 5)), index=Index([f'i-{i}' for i in range(5)], name='a'), columns=Index([f'i-{i}' for i in range(5)], name='a'))
    stacked = df.stack(future_stack=True)
    idx = stacked.index
    with pytest.raises(TypeError, match='^Level type mismatch'):
        idx.slice_locs(timedelta(seconds=30))
    with pytest.raises(TypeError, match='^Level type mismatch'):
        idx.slice_locs(df.index[1], (16, 'a'))