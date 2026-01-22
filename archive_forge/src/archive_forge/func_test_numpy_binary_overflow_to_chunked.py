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
@pytest.mark.slow
@pytest.mark.large_memory
def test_numpy_binary_overflow_to_chunked():
    values = [b'x']
    unicode_values = ['x']
    unique_strings = {i: b'x' * ((1 << 20) - 1) + str(i % 10).encode('utf8') for i in range(10)}
    unicode_unique_strings = {i: x.decode('utf8') for i, x in unique_strings.items()}
    values += [unique_strings[i % 10] for i in range(1 << 11)]
    unicode_values += [unicode_unique_strings[i % 10] for i in range(1 << 11)]
    for case, ex_type in [(values, pa.binary()), (unicode_values, pa.utf8())]:
        arr = np.array(case)
        arrow_arr = pa.array(arr)
        arr = None
        assert isinstance(arrow_arr, pa.ChunkedArray)
        assert arrow_arr.type == ex_type
        assert arrow_arr.num_chunks == 129
        value_index = 0
        for i in range(arrow_arr.num_chunks):
            chunk = arrow_arr.chunk(i)
            for val in chunk:
                assert val.as_py() == case[value_index]
                value_index += 1