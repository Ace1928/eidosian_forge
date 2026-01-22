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
@pytest.mark.parametrize('key_type', [iter, np.array, Series, Index])
def test_loc_getitem_iterable(self, float_frame, key_type):
    idx = key_type(['A', 'B', 'C'])
    result = float_frame.loc[:, idx]
    expected = float_frame.loc[:, ['A', 'B', 'C']]
    tm.assert_frame_equal(result, expected)