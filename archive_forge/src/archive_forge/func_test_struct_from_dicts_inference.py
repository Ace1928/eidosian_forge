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
def test_struct_from_dicts_inference():
    expected_type = pa.struct([pa.field('a', pa.int64()), pa.field('b', pa.string()), pa.field('c', pa.bool_())])
    data = [{'a': 5, 'b': 'foo', 'c': True}, {'a': 6, 'b': 'bar', 'c': False}]
    arr = pa.array(data)
    check_struct_type(arr.type, expected_type)
    assert arr.to_pylist() == data
    data = [{'a': 5, 'c': True}, None, {}, {'a': None, 'b': 'bar'}]
    expected = [{'a': 5, 'b': None, 'c': True}, None, {'a': None, 'b': None, 'c': None}, {'a': None, 'b': 'bar', 'c': None}]
    arr = pa.array(data)
    data_as_ndarray = np.empty(len(data), dtype=object)
    data_as_ndarray[:] = data
    arr2 = pa.array(data)
    check_struct_type(arr.type, expected_type)
    assert arr.to_pylist() == expected
    assert arr.equals(arr2)
    expected_type = pa.struct([pa.field('a', pa.struct([pa.field('aa', pa.list_(pa.int64())), pa.field('ab', pa.bool_())])), pa.field('b', pa.string())])
    data = [{'a': {'aa': [5, 6], 'ab': True}, 'b': 'foo'}, {'a': {'aa': None, 'ab': False}, 'b': None}, {'a': None, 'b': 'bar'}]
    arr = pa.array(data)
    assert arr.to_pylist() == data
    arr = pa.array([{}])
    assert arr.type == pa.struct([])
    assert arr.to_pylist() == [{}]
    with pytest.raises((pa.ArrowInvalid, pa.ArrowTypeError)):
        pa.array([1, {'a': 2}])