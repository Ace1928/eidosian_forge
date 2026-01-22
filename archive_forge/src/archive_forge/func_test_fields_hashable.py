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
def test_fields_hashable():
    in_dict = {}
    fields = [pa.field('a', pa.int32()), pa.field('a', pa.int64()), pa.field('a', pa.int64(), nullable=False), pa.field('b', pa.int32()), pa.field('b', pa.int32(), nullable=False)]
    for i, field in enumerate(fields):
        in_dict[field] = i
    assert len(in_dict) == len(fields)
    for i, field in enumerate(fields):
        assert in_dict[field] == i