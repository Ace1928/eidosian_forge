from copy import (
import numpy as np
import pytest
from pandas.core.dtypes.common import is_scalar
from pandas import (
import pandas._testing as tm
def test_data_deprecated(self, frame_or_series):
    obj = frame_or_series()
    msg = '(Series|DataFrame)._data is deprecated'
    with tm.assert_produces_warning(DeprecationWarning, match=msg):
        mgr = obj._data
    assert mgr is obj._mgr