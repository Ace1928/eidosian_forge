from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_combine_chunks():
    arr = pa.array([1, 2])
    chunked_arr = pa.chunked_array([arr, arr])
    res = chunked_arr.combine_chunks()
    expected = pa.array([1, 2, 1, 2])
    assert res.equals(expected)