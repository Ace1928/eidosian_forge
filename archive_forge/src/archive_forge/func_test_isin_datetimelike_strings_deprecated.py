from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
@pytest.mark.parametrize('dtype', ['m8[ns]', 'M8[ns]', 'M8[ns, UTC]'])
def test_isin_datetimelike_strings_deprecated(self, dtype):
    dta = date_range('2013-01-01', periods=3)._values
    arr = Series(dta.view('i8')).array.view(dtype)
    vals = [str(x) for x in arr]
    msg = "The behavior of 'isin' with dtype=.* is deprecated"
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res = algos.isin(arr, vals)
    assert res.all()
    vals2 = np.array(vals, dtype=str)
    with tm.assert_produces_warning(FutureWarning, match=msg):
        res2 = algos.isin(arr, vals2)
    assert res2.all()