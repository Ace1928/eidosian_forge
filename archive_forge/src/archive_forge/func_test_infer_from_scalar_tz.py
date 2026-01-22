from datetime import (
import numpy as np
import pytest
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import is_dtype_equal
from pandas import (
@pytest.mark.parametrize('tz', ['UTC', 'US/Eastern', 'Asia/Tokyo'])
def test_infer_from_scalar_tz(tz):
    dt = Timestamp(1, tz=tz)
    dtype, val = infer_dtype_from_scalar(dt)
    exp_dtype = f'datetime64[ns, {tz}]'
    assert dtype == exp_dtype
    assert val == dt