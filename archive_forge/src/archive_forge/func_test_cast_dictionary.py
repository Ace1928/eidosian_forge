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
def test_cast_dictionary():
    arr = pa.array(['foo', 'bar', None], type=pa.dictionary(pa.int64(), pa.string()))
    expected = pa.array(['foo', 'bar', None])
    assert arr.type == pa.dictionary(pa.int64(), pa.string())
    assert arr.cast(pa.string()) == expected
    for key_type in [pa.int8(), pa.int16(), pa.int32()]:
        typ = pa.dictionary(key_type, pa.string())
        expected = pa.array(['foo', 'bar', None], type=pa.dictionary(key_type, pa.string()))
        assert arr.cast(typ) == expected
    with pytest.raises(pa.ArrowInvalid):
        arr.cast(pa.int32())