import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index', [Index(np.arange(5), dtype=np.int64), RangeIndex(5)])
def test_scalar_integer_contains_float(self, index, frame_or_series):
    obj = gen_obj(frame_or_series, index)
    assert 3.0 in obj