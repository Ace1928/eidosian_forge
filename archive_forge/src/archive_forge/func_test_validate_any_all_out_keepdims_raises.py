import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
@pytest.mark.parametrize('func', [np.any, np.all])
@pytest.mark.parametrize('kwargs', [{'keepdims': True}, {'out': object()}])
def test_validate_any_all_out_keepdims_raises(kwargs, func):
    ser = Series([1, 2])
    param = next(iter(kwargs))
    name = func.__name__
    msg = f"the '{param}' parameter is not supported in the pandas implementation of {name}\\(\\)"
    with pytest.raises(ValueError, match=msg):
        func(ser, **kwargs)