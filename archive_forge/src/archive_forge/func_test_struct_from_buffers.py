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
def test_struct_from_buffers():
    ty = pa.struct([pa.field('a', pa.int16()), pa.field('b', pa.utf8())])
    array = pa.array([{'a': 0, 'b': 'foo'}, None, {'a': 5, 'b': ''}], type=ty)
    buffers = array.buffers()
    with pytest.raises(ValueError):
        pa.Array.from_buffers(ty, 3, [None, buffers[1]])
    children = [pa.Array.from_buffers(pa.int16(), 3, buffers[1:3]), pa.Array.from_buffers(pa.utf8(), 3, buffers[3:])]
    copied = pa.Array.from_buffers(ty, 3, buffers[:1], children=children)
    assert copied.equals(array)
    with pytest.raises(ValueError):
        pa.Array.from_buffers(ty, 3, [buffers[0]], children=children[:1])