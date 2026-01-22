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
def test_is_binary_string():
    assert types.is_binary(pa.binary())
    assert not types.is_binary(pa.string())
    assert not types.is_binary(pa.large_binary())
    assert not types.is_binary(pa.large_string())
    assert types.is_string(pa.string())
    assert types.is_unicode(pa.string())
    assert not types.is_string(pa.binary())
    assert not types.is_string(pa.large_string())
    assert not types.is_string(pa.large_binary())
    assert types.is_large_binary(pa.large_binary())
    assert not types.is_large_binary(pa.large_string())
    assert not types.is_large_binary(pa.binary())
    assert not types.is_large_binary(pa.string())
    assert types.is_large_string(pa.large_string())
    assert not types.is_large_string(pa.large_binary())
    assert not types.is_large_string(pa.string())
    assert not types.is_large_string(pa.binary())
    assert types.is_fixed_size_binary(pa.binary(5))
    assert not types.is_fixed_size_binary(pa.binary())