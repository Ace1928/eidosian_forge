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
def test_cast_signed_to_unsigned():
    safe_cases = [(np.array([0, 1, 2, 3], dtype='i1'), pa.uint8(), np.array([0, 1, 2, 3], dtype='u1'), pa.uint8()), (np.array([0, 1, 2, 3], dtype='i2'), pa.uint16(), np.array([0, 1, 2, 3], dtype='u2'), pa.uint16())]
    for case in safe_cases:
        _check_cast_case(case)