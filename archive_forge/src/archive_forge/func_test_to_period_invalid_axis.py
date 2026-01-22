import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_to_period_invalid_axis(self):
    dr = date_range('1/1/2000', '1/1/2001')
    df = DataFrame(np.random.default_rng(2).standard_normal((len(dr), 5)), index=dr)
    df['mix'] = 'a'
    msg = 'No axis named 2 for object type DataFrame'
    with pytest.raises(ValueError, match=msg):
        df.to_period(axis=2)