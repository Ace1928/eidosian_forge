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
def test_frame_loc_getitem_callable_mixture(self):
    df = DataFrame({'A': [1, 2, 3, 4], 'B': list('aabb'), 'C': [1, 2, 3, 4]})
    res = df.loc[lambda x: x.A > 2, ['A', 'B']]
    tm.assert_frame_equal(res, df.loc[df.A > 2, ['A', 'B']])
    res = df.loc[[2, 3], lambda x: ['A', 'B']]
    tm.assert_frame_equal(res, df.loc[[2, 3], ['A', 'B']])
    res = df.loc[3, lambda x: ['A', 'B']]
    tm.assert_series_equal(res, df.loc[3, ['A', 'B']])