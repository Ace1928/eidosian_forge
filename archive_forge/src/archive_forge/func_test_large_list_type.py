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
def test_large_list_type():
    ty = pa.large_list(pa.utf8())
    assert isinstance(ty, pa.LargeListType)
    assert ty.value_type == pa.utf8()
    assert ty.value_field == pa.field('item', pa.utf8(), nullable=True)
    with pytest.raises(TypeError):
        pa.large_list(None)