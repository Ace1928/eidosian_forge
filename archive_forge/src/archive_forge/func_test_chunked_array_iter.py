from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_chunked_array_iter():
    data = [pa.array([0]), pa.array([1, 2, 3]), pa.array([4, 5, 6]), pa.array([7, 8, 9])]
    arr = pa.chunked_array(data)
    for i, j in zip(range(10), arr):
        assert i == j.as_py()
    assert isinstance(arr, Iterable)