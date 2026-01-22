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
@pytest.mark.parametrize(('list_array_type', 'list_type_factory'), [(pa.ListArray, pa.list_), (pa.LargeListArray, pa.large_list)])
def test_list_from_arrays(list_array_type, list_type_factory):
    offsets_arr = np.array([0, 2, 5, 8], dtype='i4')
    offsets = pa.array(offsets_arr, type='int32')
    pyvalues = [b'a', b'b', b'c', b'd', b'e', b'f', b'g', b'h']
    values = pa.array(pyvalues, type='binary')
    result = list_array_type.from_arrays(offsets, values)
    expected = pa.array([pyvalues[:2], pyvalues[2:5], pyvalues[5:8]], type=list_type_factory(pa.binary()))
    assert result.equals(expected)
    typ = list_type_factory(pa.field('name', pa.binary()))
    result = list_array_type.from_arrays(offsets, values, typ)
    assert result.type == typ
    assert result.type.value_field.name == 'name'
    offsets = [0, None, 2, 6]
    values = [b'a', b'b', b'c', b'd', b'e', b'f']
    result = list_array_type.from_arrays(offsets, values)
    expected = pa.array([values[:2], None, values[2:]], type=list_type_factory(pa.binary()))
    assert result.equals(expected)
    offsets2 = [0, 2, None, 6]
    result = list_array_type.from_arrays(offsets2, values)
    expected = pa.array([values[:2], values[2:], None], type=list_type_factory(pa.binary()))
    assert result.equals(expected)
    offsets = [1, 3, 10]
    values = np.arange(5)
    with pytest.raises(ValueError):
        list_array_type.from_arrays(offsets, values)
    offsets = [0, 3, 2, 6]
    values = list(range(6))
    result = list_array_type.from_arrays(offsets, values)
    with pytest.raises(ValueError):
        result.validate(full=True)
    typ = list_type_factory(pa.binary())
    with pytest.raises(TypeError):
        list_array_type.from_arrays(offsets, values, type=typ)