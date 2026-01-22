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
def test_reindex_copies_ea(self, using_copy_on_write):
    N = 10
    df = DataFrame(np.random.default_rng(2).standard_normal((N * 10, N)), dtype='Float64')
    cols = np.arange(N)
    np.random.default_rng(2).shuffle(cols)
    result = df.reindex(columns=cols, copy=True)
    if using_copy_on_write:
        assert np.shares_memory(result[0].array._data, df[0].array._data)
    else:
        assert not np.shares_memory(result[0].array._data, df[0].array._data)
    result2 = df.reindex(columns=cols, index=df.index, copy=True)
    if using_copy_on_write:
        assert np.shares_memory(result2[0].array._data, df[0].array._data)
    else:
        assert not np.shares_memory(result2[0].array._data, df[0].array._data)