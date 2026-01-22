from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.base import _registry as ea_registry
from pandas.core.dtypes.common import is_object_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tseries.offsets import BDay
def test_setitem_list_not_dataframe(self, float_frame):
    data = np.random.default_rng(2).standard_normal((len(float_frame), 2))
    float_frame[['A', 'B']] = data
    tm.assert_almost_equal(float_frame[['A', 'B']].values, data)