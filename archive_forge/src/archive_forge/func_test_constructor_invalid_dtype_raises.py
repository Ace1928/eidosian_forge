from __future__ import annotations
from datetime import (
from functools import partial
from operator import attrgetter
import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
@pytest.mark.parametrize('dtype', [object, np.int32, np.int64])
def test_constructor_invalid_dtype_raises(self, dtype):
    msg = "Unexpected value for 'dtype'"
    with pytest.raises(ValueError, match=msg):
        DatetimeIndex([1, 2], dtype=dtype)