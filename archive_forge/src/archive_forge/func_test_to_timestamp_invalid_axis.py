from datetime import timedelta
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_to_timestamp_invalid_axis(self):
    index = period_range(freq='Y', start='1/1/2001', end='12/1/2009')
    obj = DataFrame(np.random.default_rng(2).standard_normal((len(index), 5)), index=index)
    with pytest.raises(ValueError, match='axis'):
        obj.to_timestamp(axis=2)