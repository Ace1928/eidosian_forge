from datetime import (
import numpy as np
import pytest
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import is_dtype_equal
from pandas import (
@pytest.mark.parametrize('freq', ['M', 'D'])
def test_infer_dtype_from_period(freq):
    p = Period('2011-01-01', freq=freq)
    dtype, val = infer_dtype_from_scalar(p)
    exp_dtype = f'period[{freq}]'
    assert dtype == exp_dtype
    assert val == p