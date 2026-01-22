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
def test_loc_slice_disallows_positional():
    dti = date_range('2016-01-01', periods=3)
    df = DataFrame(np.random.default_rng(2).random((3, 2)), index=dti)
    ser = df[0]
    msg = 'cannot do slice indexing on DatetimeIndex with these indexers \\[1\\] of type int'
    for obj in [df, ser]:
        with pytest.raises(TypeError, match=msg):
            obj.loc[1:3]
        with pytest.raises(TypeError, match='Slicing a positional slice with .loc'):
            obj.loc[1:3] = 1
    with pytest.raises(TypeError, match=msg):
        df.loc[1:3, 1]
    with pytest.raises(TypeError, match='Slicing a positional slice with .loc'):
        df.loc[1:3, 1] = 2