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
@pytest.mark.parametrize('list_type', [pa.list_(pa.int32()), pa.list_(pa.int32(), list_size=2), pa.large_list(pa.int32())])
def test_list_value_parent_indices(list_type):
    arr = pa.array([[0, 1], None, [None, None], [3, 4]], type=list_type)
    expected = pa.array([0, 0, 2, 2, 3, 3], type=pa.int64())
    assert arr.value_parent_indices().equals(expected)