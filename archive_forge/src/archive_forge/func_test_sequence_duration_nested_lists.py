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
def test_sequence_duration_nested_lists():
    td1 = datetime.timedelta(1, 1, 1000)
    td2 = datetime.timedelta(1, 100)
    data = [[td1, None], [td1, td2]]
    arr = pa.array(data)
    assert len(arr) == 2
    assert arr.type == pa.list_(pa.duration('us'))
    assert arr.to_pylist() == data
    arr = pa.array(data, type=pa.list_(pa.duration('ms')))
    assert len(arr) == 2
    assert arr.type == pa.list_(pa.duration('ms'))
    assert arr.to_pylist() == data