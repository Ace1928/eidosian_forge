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
def test_struct_duplicate_field_names():
    fields = [pa.field('a', pa.int64()), pa.field('b', pa.int32()), pa.field('a', pa.int32())]
    ty = pa.struct(fields)
    with pytest.warns(UserWarning):
        with pytest.raises(KeyError):
            ty['a']
    assert ty.get_field_index('a') == -1
    assert ty.get_all_field_indices('a') == [0, 2]