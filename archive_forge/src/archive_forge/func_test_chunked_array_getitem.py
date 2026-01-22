from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_chunked_array_getitem():
    data = [pa.array([1, 2, 3]), pa.array([4, 5, 6])]
    data = pa.chunked_array(data)
    assert data[1].as_py() == 2
    assert data[-1].as_py() == 6
    assert data[-6].as_py() == 1
    with pytest.raises(IndexError):
        data[6]
    with pytest.raises(IndexError):
        data[-7]
    assert data[np.int32(1)].as_py() == 2
    data_slice = data[2:4]
    assert data_slice.to_pylist() == [3, 4]
    data_slice = data[4:-1]
    assert data_slice.to_pylist() == [5]
    data_slice = data[99:99]
    assert data_slice.type == data.type
    assert data_slice.to_pylist() == []