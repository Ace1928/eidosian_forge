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
def test_reindex_nearest_tz(self, tz_aware_fixture):
    tz = tz_aware_fixture
    idx = date_range('2019-01-01', periods=5, tz=tz)
    df = DataFrame({'x': list(range(5))}, index=idx)
    expected = df.head(3)
    actual = df.reindex(idx[:3], method='nearest')
    tm.assert_frame_equal(expected, actual)