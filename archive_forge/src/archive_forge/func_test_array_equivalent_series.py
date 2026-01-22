from contextlib import nullcontext
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas._config import config as cf
from pandas._libs import missing as libmissing
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('val', [1, 1.1, 1 + 1j, True, 'abc', [1, 2], (1, 2), {1, 2}, {'a': 1}, None])
def test_array_equivalent_series(val):
    arr = np.array([1, 2])
    msg = 'elementwise comparison failed'
    cm = tm.assert_produces_warning(FutureWarning, match=msg, check_stacklevel=False) if isinstance(val, str) and (not np_version_gte1p25) else nullcontext()
    with cm:
        assert not array_equivalent(Series([arr, arr]), Series([arr, val]))