from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
def test_isin_nan_common_float64(self, nulls_fixture, float_numpy_dtype):
    dtype = float_numpy_dtype
    if nulls_fixture is pd.NaT or nulls_fixture is pd.NA:
        msg = f'float\\(\\) argument must be a string or a (real )?number, not {repr(type(nulls_fixture).__name__)}'
        with pytest.raises(TypeError, match=msg):
            Index([1.0, nulls_fixture], dtype=dtype)
        idx = Index([1.0, np.nan], dtype=dtype)
        assert not idx.isin([nulls_fixture]).any()
        return
    idx = Index([1.0, nulls_fixture], dtype=dtype)
    res = idx.isin([np.nan])
    tm.assert_numpy_array_equal(res, np.array([False, True]))
    res = idx.isin([pd.NaT])
    tm.assert_numpy_array_equal(res, np.array([False, False]))