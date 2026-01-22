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
def test_dti_set_index_reindex_datetimeindex(self):
    df = DataFrame(np.random.default_rng(2).random(6))
    idx1 = date_range('2011/01/01', periods=6, freq='ME', tz='US/Eastern')
    idx2 = date_range('2013', periods=6, freq='YE', tz='Asia/Tokyo')
    df = df.set_index(idx1)
    tm.assert_index_equal(df.index, idx1)
    df = df.reindex(idx2)
    tm.assert_index_equal(df.index, idx2)