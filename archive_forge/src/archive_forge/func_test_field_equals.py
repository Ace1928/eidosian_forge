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
def test_field_equals():
    meta1 = {b'foo': b'bar'}
    meta2 = {b'bizz': b'bazz'}
    f1 = pa.field('a', pa.int8(), nullable=True)
    f2 = pa.field('a', pa.int8(), nullable=True)
    f3 = pa.field('a', pa.int8(), nullable=False)
    f4 = pa.field('a', pa.int16(), nullable=False)
    f5 = pa.field('b', pa.int16(), nullable=False)
    f6 = pa.field('a', pa.int8(), nullable=True, metadata=meta1)
    f7 = pa.field('a', pa.int8(), nullable=True, metadata=meta1)
    f8 = pa.field('a', pa.int8(), nullable=True, metadata=meta2)
    assert f1.equals(f2)
    assert f6.equals(f7)
    assert not f1.equals(f3)
    assert not f1.equals(f4)
    assert not f3.equals(f4)
    assert not f4.equals(f5)
    assert f1.equals(f6)
    assert not f1.equals(f6, check_metadata=True)
    assert f6.equals(f7)
    assert f7.equals(f8)
    assert not f7.equals(f8, check_metadata=True)