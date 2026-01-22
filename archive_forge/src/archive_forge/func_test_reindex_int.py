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
def test_reindex_int(self, int_frame):
    smaller = int_frame.reindex(int_frame.index[::2])
    assert smaller['A'].dtype == np.int64
    bigger = smaller.reindex(int_frame.index)
    assert bigger['A'].dtype == np.float64
    smaller = int_frame.reindex(columns=['A', 'B'])
    assert smaller['A'].dtype == np.int64