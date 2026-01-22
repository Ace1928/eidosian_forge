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
def test_loc_indexer_length_one(self):
    df = DataFrame({'a': ['x'], 'b': ['y']}, dtype=object)
    expected = DataFrame({'a': ['y'], 'b': ['y']}, dtype=object)
    df.loc[np.array([True], dtype=np.bool_), ['a']] = df['b'].copy()
    tm.assert_frame_equal(df, expected)