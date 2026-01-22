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
@pytest.mark.parametrize('na', (np.nan, pd.NA, None, pd.NaT))
def test_loc_consistency_series_enlarge_set_into(self, na):
    srs_enlarge = Series(['a', 'b', 'c'], dtype='category')
    srs_enlarge.loc[3] = na
    srs_setinto = Series(['a', 'b', 'c', 'a'], dtype='category')
    srs_setinto.loc[3] = na
    tm.assert_series_equal(srs_enlarge, srs_setinto)
    expected = Series(['a', 'b', 'c', na], dtype='category')
    tm.assert_series_equal(srs_enlarge, expected)