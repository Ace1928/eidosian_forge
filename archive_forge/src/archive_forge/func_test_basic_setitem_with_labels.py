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
def test_basic_setitem_with_labels(self, datetime_series):
    indices = datetime_series.index[[5, 10, 15]]
    cp = datetime_series.copy()
    exp = datetime_series.copy()
    cp[indices] = 0
    exp.loc[indices] = 0
    tm.assert_series_equal(cp, exp)
    cp = datetime_series.copy()
    exp = datetime_series.copy()
    cp[indices[0]:indices[2]] = 0
    exp.loc[indices[0]:indices[2]] = 0
    tm.assert_series_equal(cp, exp)