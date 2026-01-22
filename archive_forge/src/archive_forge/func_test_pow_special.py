from datetime import (
import pickle
import numpy as np
import pytest
from pandas._libs.missing import NA
from pandas.core.dtypes.common import is_scalar
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('value', [0, 0.0, -0, -0.0, False, np.bool_(False), np.int_(0), np.float64(0), np.int_(-0), np.float64(-0)])
@pytest.mark.parametrize('asarray', [True, False])
def test_pow_special(value, asarray):
    if asarray:
        value = np.array([value])
    result = NA ** value
    if asarray:
        result = result[0]
    else:
        assert isinstance(result, type(value))
    assert result == 1