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
def test_cast_list_to_primitive():
    arr = pa.array([[1, 2], [3, 4]])
    with pytest.raises(NotImplementedError):
        arr.cast(pa.int8())
    arr = pa.array([[b'a', b'b'], [b'c']], pa.list_(pa.binary()))
    with pytest.raises(NotImplementedError):
        arr.cast(pa.binary())