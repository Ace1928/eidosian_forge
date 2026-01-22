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
@pytest.mark.parametrize('val', [0, np.array(0, dtype=np.int64), np.array([0, 0, 0, 0, 0], dtype=np.int64)])
def test_loc_setitem_consistency(self, frame_for_consistency, val):
    expected = DataFrame({'date': Series(0, index=range(5), dtype=np.int64), 'val': Series(range(5), dtype=np.int64)})
    df = frame_for_consistency.copy()
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        df.loc[:, 'date'] = val
    tm.assert_frame_equal(df, expected)