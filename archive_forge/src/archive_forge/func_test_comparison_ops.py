from datetime import (
import pickle
import numpy as np
import pytest
from pandas._libs.missing import NA
from pandas.core.dtypes.common import is_scalar
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('other', [NA, 1, 1.0, 'a', b'a', np.int64(1), np.nan, np.bool_(True), time(0), date(1, 2, 3), timedelta(1), pd.NaT])
def test_comparison_ops(comparison_op, other):
    assert comparison_op(NA, other) is NA
    assert comparison_op(other, NA) is NA