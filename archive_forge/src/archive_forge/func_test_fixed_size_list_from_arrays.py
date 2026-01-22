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
def test_fixed_size_list_from_arrays():
    values = pa.array(range(12), pa.int64())
    result = pa.FixedSizeListArray.from_arrays(values, 4)
    assert result.to_pylist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    assert result.type.equals(pa.list_(pa.int64(), 4))
    typ = pa.list_(pa.field('name', pa.int64()), 4)
    result = pa.FixedSizeListArray.from_arrays(values, type=typ)
    assert result.to_pylist() == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
    assert result.type.equals(typ)
    assert result.type.value_field.name == 'name'
    result = pa.FixedSizeListArray.from_arrays(values, type=typ, mask=pa.array([False, True, False]))
    assert result.to_pylist() == [[0, 1, 2, 3], None, [8, 9, 10, 11]]
    result = pa.FixedSizeListArray.from_arrays(values, list_size=4, mask=pa.array([False, True, False]))
    assert result.to_pylist() == [[0, 1, 2, 3], None, [8, 9, 10, 11]]
    with pytest.raises(ValueError):
        pa.FixedSizeListArray.from_arrays(values, -4)
    with pytest.raises(ValueError):
        pa.FixedSizeListArray.from_arrays(pa.array([], pa.int64()), 0)
    with pytest.raises(ValueError):
        pa.FixedSizeListArray.from_arrays(values, 5)
    typ = pa.list_(pa.int64(), 5)
    with pytest.raises(ValueError):
        pa.FixedSizeListArray.from_arrays(values, type=typ)
    typ = pa.list_(pa.float64(), 4)
    with pytest.raises(TypeError):
        pa.FixedSizeListArray.from_arrays(values, type=typ)
    with pytest.raises(ValueError):
        pa.FixedSizeListArray.from_arrays(values)
    typ = pa.list_(pa.int64(), 4)
    with pytest.raises(ValueError):
        pa.FixedSizeListArray.from_arrays(values, list_size=4, type=typ)