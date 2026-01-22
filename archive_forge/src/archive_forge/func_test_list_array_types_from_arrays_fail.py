from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
@pytest.mark.parametrize(('list_array_type', 'list_type_factory'), ((pa.ListArray, pa.list_), (pa.LargeListArray, pa.large_list)))
def test_list_array_types_from_arrays_fail(list_array_type, list_type_factory):
    arr = pa.array([[0], None, [0, None], [0]], list_type_factory(pa.int8()))
    offsets = pa.array([0, None, 1, 3, 4])
    reconstructed_arr = list_array_type.from_arrays(arr.offsets, arr.values)
    assert reconstructed_arr.to_pylist() == [[0], [], [0, None], [0]]
    reconstructed_arr = list_array_type.from_arrays(offsets, arr.values)
    assert arr == reconstructed_arr
    reconstructed_arr = list_array_type.from_arrays(arr.offsets, arr.values, mask=arr.is_null())
    assert arr == reconstructed_arr
    with pytest.raises(ValueError, match='Ambiguous to specify both '):
        list_array_type.from_arrays(offsets, arr.values, mask=arr.is_null())
    arr_slice = arr[1:]
    msg = 'Null bitmap with offsets slice not supported.'
    with pytest.raises(NotImplementedError, match=msg):
        list_array_type.from_arrays(arr_slice.offsets, arr_slice.values, mask=arr_slice.is_null())