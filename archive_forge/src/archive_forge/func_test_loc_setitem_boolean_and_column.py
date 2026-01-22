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
def test_loc_setitem_boolean_and_column(self, float_frame):
    expected = float_frame.copy()
    mask = float_frame['A'] > 0
    float_frame.loc[mask, 'B'] = 0
    values = expected.values.copy()
    values[mask.values, 1] = 0
    expected = DataFrame(values, index=expected.index, columns=expected.columns)
    tm.assert_frame_equal(float_frame, expected)