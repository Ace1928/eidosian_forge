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
def test_nested_ndarray_in_object_array():
    arr = np.empty(2, dtype=object)
    arr[:] = [np.array([1, 2], dtype=np.int64), np.array([2, 3], dtype=np.int64)]
    arr2 = np.empty(2, dtype=object)
    arr2[0] = [3, 4]
    arr2[1] = [5, 6]
    expected_type = pa.list_(pa.list_(pa.int64()))
    assert pa.infer_type([arr]) == expected_type
    result = pa.array([arr, arr2])
    expected = pa.array([[[1, 2], [2, 3]], [[3, 4], [5, 6]]], type=expected_type)
    assert result.equals(expected)
    arr = np.empty(2, dtype=object)
    arr[:] = [np.array([1]), np.array([2])]
    result = pa.array([arr, arr])
    assert result.to_pylist() == [[[1], [2]], [[1], [2]]]