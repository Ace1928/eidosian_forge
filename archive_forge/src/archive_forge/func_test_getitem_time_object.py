from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
def test_getitem_time_object(self):
    rng = date_range('1/1/2000', '1/5/2000', freq='5min')
    ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    mask = (rng.hour == 9) & (rng.minute == 30)
    result = ts[time(9, 30)]
    expected = ts[mask]
    result.index = result.index._with_freq(None)
    tm.assert_series_equal(result, expected)