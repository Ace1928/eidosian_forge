from collections import OrderedDict
from collections.abc import Iterator
from functools import partial
import datetime
import sys
import pytest
import hypothesis as h
import hypothesis.strategies as st
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.types as types
import pyarrow.tests.strategies as past
def test_bit_width():
    for ty, expected in [(pa.bool_(), 1), (pa.int8(), 8), (pa.uint32(), 32), (pa.float16(), 16), (pa.decimal128(19, 4), 128), (pa.decimal256(76, 38), 256), (pa.binary(42), 42 * 8)]:
        assert ty.bit_width == expected
    for ty in [pa.binary(), pa.string(), pa.list_(pa.int16())]:
        with pytest.raises(ValueError, match='fixed width'):
            ty.bit_width