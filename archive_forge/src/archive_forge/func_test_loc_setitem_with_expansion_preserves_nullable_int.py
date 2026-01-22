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
@pytest.mark.parametrize('dtype', ['Int32', 'Int64', 'UInt32', 'UInt64', 'Float32', 'Float64'])
def test_loc_setitem_with_expansion_preserves_nullable_int(self, dtype):
    ser = Series([0, 1, 2, 3], dtype=dtype)
    df = DataFrame({'data': ser})
    result = DataFrame(index=df.index)
    result.loc[df.index, 'data'] = ser
    tm.assert_frame_equal(result, df, check_column_type=False)
    result = DataFrame(index=df.index)
    result.loc[df.index, 'data'] = ser._values
    tm.assert_frame_equal(result, df, check_column_type=False)