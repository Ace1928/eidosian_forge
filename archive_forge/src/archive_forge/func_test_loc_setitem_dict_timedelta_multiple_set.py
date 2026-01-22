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
@td.skip_array_manager_invalid_test
def test_loc_setitem_dict_timedelta_multiple_set(self):
    result = DataFrame(columns=['time', 'value'])
    result.loc[1] = {'time': Timedelta(6, unit='s'), 'value': 'foo'}
    result.loc[1] = {'time': Timedelta(6, unit='s'), 'value': 'foo'}
    expected = DataFrame([[Timedelta(6, unit='s'), 'foo']], columns=['time', 'value'], index=[1])
    tm.assert_frame_equal(result, expected)