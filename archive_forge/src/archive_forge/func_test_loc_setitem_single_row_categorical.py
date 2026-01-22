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
def test_loc_setitem_single_row_categorical(self, using_infer_string):
    df = DataFrame({'Alpha': ['a'], 'Numeric': [0]})
    categories = Categorical(df['Alpha'], categories=['a', 'b', 'c'])
    df.loc[:, 'Alpha'] = categories
    result = df['Alpha']
    expected = Series(categories, index=df.index, name='Alpha').astype(object if not using_infer_string else 'string[pyarrow_numpy]')
    tm.assert_series_equal(result, expected)
    df['Alpha'] = categories
    tm.assert_series_equal(df['Alpha'], Series(categories, name='Alpha'))