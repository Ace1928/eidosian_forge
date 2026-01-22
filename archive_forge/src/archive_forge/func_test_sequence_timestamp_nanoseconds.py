import collections
import datetime
import decimal
import itertools
import math
import re
import sys
import hypothesis as h
import numpy as np
import pytest
from pyarrow.pandas_compat import _pandas_api  # noqa
import pyarrow as pa
from pyarrow.tests import util
import pyarrow.tests.strategies as past
@pytest.mark.xfail(not _pandas_api.have_pandas, reason='pandas required for nanosecond conversion')
def test_sequence_timestamp_nanoseconds():
    inputs = [[datetime.datetime(2007, 7, 13, 1, 23, 34, 123456)], [MyDatetime(2007, 7, 13, 1, 23, 34, 123456)]]
    for data in inputs:
        ns = pa.timestamp('ns')
        arr_ns = pa.array(data, type=ns)
        assert len(arr_ns) == 1
        assert arr_ns.type == ns
        assert arr_ns[0].as_py() == datetime.datetime(2007, 7, 13, 1, 23, 34, 123456)