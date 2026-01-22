import gc
import decimal
import json
import multiprocessing as mp
import sys
import warnings
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from pyarrow.pandas_compat import get_logical_type, _pandas_api
from pyarrow.tests.util import invoke_script, random_ascii, rands
import pyarrow.tests.strategies as past
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
import pyarrow as pa
@pytest.mark.parametrize('t,data,expected', [(pa.int64, [[1, 2], [3], None], [None, [3], None]), (pa.string, [['aaa', 'bb'], ['c'], None], [None, ['c'], None]), (pa.null, [[None, None], [None], None], [None, [None], None])])
def test_array_from_pandas_typed_array_with_mask(self, t, data, expected):
    m = np.array([True, False, True])
    s = pd.Series(data)
    result = pa.Array.from_pandas(s, mask=m, type=pa.list_(t()))
    assert pa.Array.from_pandas(expected, type=pa.list_(t())).equals(result)