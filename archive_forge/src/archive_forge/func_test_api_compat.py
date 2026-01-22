from copy import (
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('func', ['sum', 'cumsum', 'any', 'var'])
def test_api_compat(self, func, frame_or_series):
    obj = construct(frame_or_series, 5)
    f = getattr(obj, func)
    assert f.__name__ == func
    assert f.__qualname__.endswith(func)