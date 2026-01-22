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
@pytest.mark.xfail(reason='Type inference for multidimensional ndarray not yet implemented', raises=AssertionError)
def test_multidimensional_ndarray_as_nested_list():
    arr = np.array([[1, 2], [2, 3]], dtype=np.int64)
    arr2 = np.array([[3, 4], [5, 6]], dtype=np.int64)
    expected_type = pa.list_(pa.list_(pa.int64()))
    assert pa.infer_type([arr]) == expected_type
    result = pa.array([arr, arr2])
    expected = pa.array([[[1, 2], [2, 3]], [[3, 4], [5, 6]]], type=expected_type)
    assert result.equals(expected)