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
def test_reindex_methods_nearest_special(self):
    df = DataFrame({'x': list(range(5))})
    target = np.array([-0.1, 0.9, 1.1, 1.5])
    expected = DataFrame({'x': [0, 1, 1, np.nan]}, index=target)
    actual = df.reindex(target, method='nearest', tolerance=0.2)
    tm.assert_frame_equal(expected, actual)
    expected = DataFrame({'x': [0, np.nan, 1, np.nan]}, index=target)
    actual = df.reindex(target, method='nearest', tolerance=[0.5, 0.01, 0.4, 0.1])
    tm.assert_frame_equal(expected, actual)