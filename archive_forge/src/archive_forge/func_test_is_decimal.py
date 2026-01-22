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
def test_is_decimal():
    decimal128 = pa.decimal128(19, 4)
    decimal256 = pa.decimal256(76, 38)
    int32 = pa.int32()
    assert types.is_decimal(decimal128)
    assert types.is_decimal(decimal256)
    assert not types.is_decimal(int32)
    assert types.is_decimal128(decimal128)
    assert not types.is_decimal128(decimal256)
    assert not types.is_decimal128(int32)
    assert not types.is_decimal256(decimal128)
    assert types.is_decimal256(decimal256)
    assert not types.is_decimal256(int32)