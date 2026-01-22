import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_nan(float_numpy_dtype):
    s = Index(range(10), dtype=float_numpy_dtype).to_series()
    assert s.get(np.nan) is None
    assert s.get(np.nan, default='Missing') == 'Missing'