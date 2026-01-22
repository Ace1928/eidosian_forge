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
@pytest.mark.parametrize(('ty', 'values'), [('bool', [True, False, True]), ('uint8', range(0, 255)), ('int8', range(0, 128)), ('uint16', range(0, 10)), ('int16', range(0, 10)), ('uint32', range(0, 10)), ('int32', range(0, 10)), ('uint64', range(0, 10)), ('int64', range(0, 10)), ('float', [0.0, 0.1, 0.2]), ('double', [0.0, 0.1, 0.2]), ('string', ['a', 'b', 'c']), ('binary', [b'a', b'b', b'c']), (pa.binary(3), [b'abc', b'bcd', b'cde'])])
def test_cast_identities(ty, values):
    arr = pa.array(values, type=ty)
    assert arr.cast(ty).equals(arr)