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
@pytest.mark.parametrize('arr', ([None, [0]], [None, [0, None], [0]], [[0], [1]]))
def test_list_array_types_from_arrays(list_array_type, list_type_factory, arr):
    arr = pa.array(arr, list_type_factory(pa.int8()))
    reconstructed_arr = list_array_type.from_arrays(arr.offsets, arr.values, mask=arr.is_null())
    assert arr == reconstructed_arr