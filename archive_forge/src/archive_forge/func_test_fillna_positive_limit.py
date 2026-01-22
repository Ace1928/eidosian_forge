import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import _check_mixed_float
@pytest.mark.parametrize('type', [int, float])
def test_fillna_positive_limit(self, type):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4))).astype(type)
    msg = 'Limit must be greater than 0'
    with pytest.raises(ValueError, match=msg):
        df.fillna(0, limit=-5)