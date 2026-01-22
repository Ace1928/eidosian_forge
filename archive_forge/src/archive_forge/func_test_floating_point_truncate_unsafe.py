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
def test_floating_point_truncate_unsafe():
    unsafe_cases = [(np.array([1.1, 2.2, 3.3], dtype='float32'), 'float32', np.array([1, 2, 3], dtype='i4'), pa.int32()), (np.array([1.1, 2.2, 3.3], dtype='float64'), 'float64', np.array([1, 2, 3], dtype='i4'), pa.int32()), (np.array([-10.1, 20.2, -30.3], dtype='float64'), 'float64', np.array([-10, 20, -30], dtype='i4'), pa.int32())]
    for case in unsafe_cases:
        with pytest.raises(pa.ArrowInvalid, match='truncated'):
            _check_cast_case(case, safe=True)
        _check_cast_case(case, safe=False)