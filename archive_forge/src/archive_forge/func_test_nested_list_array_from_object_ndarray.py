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
@pytest.mark.parametrize(('data', 'value_type'), [([[1, 2], [3]], pa.list_(pa.int64())), ([[1, 2], [3, 4]], pa.list_(pa.int64(), 2)), ([[1], [2, 3]], pa.large_list(pa.int64()))])
def test_nested_list_array_from_object_ndarray(data, value_type):
    ndarray = np.empty(len(data), dtype=object)
    ndarray[:] = [np.array(item, dtype=object) for item in data]
    ty = pa.list_(value_type)
    arr = pa.array([ndarray], type=ty)
    assert arr.type.equals(ty)
    assert arr.to_pylist() == [data]