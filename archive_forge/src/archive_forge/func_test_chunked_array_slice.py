from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_chunked_array_slice():
    data = [pa.array([1, 2, 3]), pa.array([4, 5, 6])]
    data = pa.chunked_array(data)
    data_slice = data.slice(len(data))
    assert data_slice.type == data.type
    assert data_slice.to_pylist() == []
    data_slice = data.slice(len(data) + 10)
    assert data_slice.type == data.type
    assert data_slice.to_pylist() == []
    table = pa.Table.from_arrays([data], names=['a'])
    table_slice = table.slice(len(table))
    assert len(table_slice) == 0
    table = pa.Table.from_arrays([data], names=['a'])
    table_slice = table.slice(len(table) + 10)
    assert len(table_slice) == 0