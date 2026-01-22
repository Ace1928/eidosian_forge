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
def test_nested_ndarray_different_dtypes():
    data = [np.array([1, 2, 3], dtype='int64'), None, np.array([4, 5, 6], dtype='uint32')]
    arr = pa.array(data)
    expected = pa.array([[1, 2, 3], None, [4, 5, 6]], type=pa.list_(pa.int64()))
    assert arr.equals(expected)
    t2 = pa.list_(pa.uint32())
    arr2 = pa.array(data, type=t2)
    expected2 = expected.cast(t2)
    assert arr2.equals(expected2)