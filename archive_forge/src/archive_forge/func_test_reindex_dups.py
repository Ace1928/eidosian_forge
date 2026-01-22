from datetime import (
import inspect
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import dateutil_gettz as gettz
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
def test_reindex_dups(self):
    arr = np.random.default_rng(2).standard_normal(10)
    df = DataFrame(arr, index=[1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
    result = df.copy()
    result.index = list(range(len(df)))
    expected = DataFrame(arr, index=list(range(len(df))))
    tm.assert_frame_equal(result, expected)
    msg = 'cannot reindex on an axis with duplicate labels'
    with pytest.raises(ValueError, match=msg):
        df.reindex(index=list(range(len(df))))