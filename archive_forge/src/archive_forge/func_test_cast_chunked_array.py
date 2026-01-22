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
def test_cast_chunked_array():
    arrays = [pa.array([1, 2, 3]), pa.array([4, 5, 6])]
    carr = pa.chunked_array(arrays)
    target = pa.float64()
    casted = carr.cast(target)
    expected = pa.chunked_array([x.cast(target) for x in arrays])
    assert casted.equals(expected)