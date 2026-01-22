from collections import namedtuple
from datetime import (
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_loc_getitem_not_monotonic(self, datetime_series):
    d1, d2 = datetime_series.index[[5, 15]]
    ts2 = datetime_series[::2].iloc[[1, 2, 0]]
    msg = "Timestamp\\('2000-01-10 00:00:00'\\)"
    with pytest.raises(KeyError, match=msg):
        ts2.loc[d1:d2]
    with pytest.raises(KeyError, match=msg):
        ts2.loc[d1:d2] = 0