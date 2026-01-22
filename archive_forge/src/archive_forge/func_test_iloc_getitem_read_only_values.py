from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
@pytest.mark.parametrize('indexer', [tm.loc, tm.iloc])
def test_iloc_getitem_read_only_values(self, indexer):
    rw_array = np.eye(10)
    rw_df = DataFrame(rw_array)
    ro_array = np.eye(10)
    ro_array.setflags(write=False)
    ro_df = DataFrame(ro_array)
    tm.assert_frame_equal(indexer(rw_df)[[1, 2, 3]], indexer(ro_df)[[1, 2, 3]])
    tm.assert_frame_equal(indexer(rw_df)[[1]], indexer(ro_df)[[1]])
    tm.assert_series_equal(indexer(rw_df)[1], indexer(ro_df)[1])
    tm.assert_frame_equal(indexer(rw_df)[1:3], indexer(ro_df)[1:3])