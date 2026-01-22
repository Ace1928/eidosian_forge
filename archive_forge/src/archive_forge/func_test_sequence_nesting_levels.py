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
def test_sequence_nesting_levels():
    data = [1, 2, None]
    arr = pa.array(data)
    assert arr.type == pa.int64()
    assert arr.to_pylist() == data
    data = [[1], [2], None]
    arr = pa.array(data)
    assert arr.type == pa.list_(pa.int64())
    assert arr.to_pylist() == data
    data = [[1], [2, 3, 4], [None]]
    arr = pa.array(data)
    assert arr.type == pa.list_(pa.int64())
    assert arr.to_pylist() == data
    data = [None, [[None, 1]], [[2, 3, 4], None], [None]]
    arr = pa.array(data)
    assert arr.type == pa.list_(pa.list_(pa.int64()))
    assert arr.to_pylist() == data
    exceptions = (pa.ArrowInvalid, pa.ArrowTypeError)
    with pytest.raises(exceptions):
        pa.array([1, 2, [1]])
    with pytest.raises(exceptions):
        pa.array([1, 2, []])
    with pytest.raises(exceptions):
        pa.array([[1], [2], [None, [1]]])