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
def test_category_zero_chunks(self):
    for pa_type, dtype in [(pa.string(), 'object'), (pa.int64(), 'int64')]:
        a = pa.chunked_array([], pa.dictionary(pa.int8(), pa_type))
        result = a.to_pandas()
        expected = pd.Categorical([], categories=np.array([], dtype=dtype))
        tm.assert_series_equal(pd.Series(result), pd.Series(expected))
        table = pa.table({'a': a})
        result = table.to_pandas()
        expected = pd.DataFrame({'a': expected})
        tm.assert_frame_equal(result, expected)