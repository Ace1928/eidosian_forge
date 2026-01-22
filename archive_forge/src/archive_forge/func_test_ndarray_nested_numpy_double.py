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
@pytest.mark.parametrize('from_pandas', [True, False])
@pytest.mark.parametrize('inner_seq', [np.array, list])
def test_ndarray_nested_numpy_double(from_pandas, inner_seq):
    data = np.array([inner_seq([1.0, 2.0]), inner_seq([1.0, 2.0, 3.0]), inner_seq([np.nan]), None], dtype=object)
    arr = pa.array(data, from_pandas=from_pandas)
    assert len(arr) == 4
    assert arr.null_count == 1
    assert arr.type == pa.list_(pa.float64())
    if from_pandas:
        assert arr.to_pylist() == [[1.0, 2.0], [1.0, 2.0, 3.0], [None], None]
    else:
        np.testing.assert_equal(arr.to_pylist(), [[1.0, 2.0], [1.0, 2.0, 3.0], [np.nan], None])