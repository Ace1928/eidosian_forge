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
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason="can't set int into string")
def test_loc_setitem_str_to_small_float_conversion_type(self):
    col_data = [str(np.random.default_rng(2).random() * 1e-12) for _ in range(5)]
    result = DataFrame(col_data, columns=['A'])
    expected = DataFrame(col_data, columns=['A'], dtype=object)
    tm.assert_frame_equal(result, expected)
    result.loc[result.index, 'A'] = [float(x) for x in col_data]
    expected = DataFrame(col_data, columns=['A'], dtype=float).astype(object)
    tm.assert_frame_equal(result, expected)
    result['A'] = [float(x) for x in col_data]
    expected = DataFrame(col_data, columns=['A'], dtype=float)
    tm.assert_frame_equal(result, expected)