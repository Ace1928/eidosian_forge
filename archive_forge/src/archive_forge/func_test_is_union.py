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
def test_is_union():
    for mode in [pa.lib.UnionMode_SPARSE, pa.lib.UnionMode_DENSE]:
        assert types.is_union(pa.union([pa.field('a', pa.int32()), pa.field('b', pa.int8()), pa.field('c', pa.string())], mode=mode))
    assert not types.is_union(pa.list_(pa.int32()))