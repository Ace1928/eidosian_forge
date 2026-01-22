from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_chunked_array_flatten():
    ty = pa.struct([pa.field('x', pa.int16()), pa.field('y', pa.float32())])
    a = pa.array([(1, 2.5), (3, 4.5), (5, 6.5)], type=ty)
    carr = pa.chunked_array(a)
    x, y = carr.flatten()
    assert x.equals(pa.chunked_array(pa.array([1, 3, 5], type=pa.int16())))
    assert y.equals(pa.chunked_array(pa.array([2.5, 4.5, 6.5], type=pa.float32())))
    a = pa.array([], type=ty)
    carr = pa.chunked_array(a)
    x, y = carr.flatten()
    assert x.equals(pa.chunked_array(pa.array([], type=pa.int16())))
    assert y.equals(pa.chunked_array(pa.array([], type=pa.float32())))