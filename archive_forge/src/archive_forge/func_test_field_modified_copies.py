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
def test_field_modified_copies():
    f0 = pa.field('foo', pa.int32(), True)
    f0_ = pa.field('foo', pa.int32(), True)
    assert f0.equals(f0_)
    f1 = pa.field('foo', pa.int64(), True)
    f1_ = f0.with_type(pa.int64())
    assert f1.equals(f1_)
    assert f0.equals(f0_)
    f2 = pa.field('foo', pa.int32(), False)
    f2_ = f0.with_nullable(False)
    assert f2.equals(f2_)
    assert f0.equals(f0_)
    f3 = pa.field('bar', pa.int32(), True)
    f3_ = f0.with_name('bar')
    assert f3.equals(f3_)
    assert f0.equals(f0_)