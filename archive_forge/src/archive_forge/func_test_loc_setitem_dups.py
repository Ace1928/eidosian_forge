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
def test_loc_setitem_dups(self):
    df_orig = DataFrame({'me': list('rttti'), 'foo': list('aaade'), 'bar': np.arange(5, dtype='float64') * 1.34 + 2, 'bar2': np.arange(5, dtype='float64') * -0.34 + 2}).set_index('me')
    indexer = ('r', ['bar', 'bar2'])
    df = df_orig.copy()
    df.loc[indexer] *= 2.0
    tm.assert_series_equal(df.loc[indexer], 2.0 * df_orig.loc[indexer])
    indexer = ('r', 'bar')
    df = df_orig.copy()
    df.loc[indexer] *= 2.0
    assert df.loc[indexer] == 2.0 * df_orig.loc[indexer]
    indexer = ('t', ['bar', 'bar2'])
    df = df_orig.copy()
    df.loc[indexer] *= 2.0
    tm.assert_frame_equal(df.loc[indexer], 2.0 * df_orig.loc[indexer])