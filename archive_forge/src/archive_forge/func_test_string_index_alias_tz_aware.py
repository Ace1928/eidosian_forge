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
@pytest.mark.parametrize('tz', ['US/Eastern', 'dateutil/US/Eastern'])
def test_string_index_alias_tz_aware(self, tz):
    rng = date_range('1/1/2000', periods=10, tz=tz)
    ser = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)
    result = ser['1/3/2000']
    tm.assert_almost_equal(result, ser.iloc[2])