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
def test_struct_type():
    fields = [pa.field('a', pa.int64()), pa.field('a', pa.int32()), pa.field('b', pa.int32())]
    ty = pa.struct(fields)
    assert len(ty) == ty.num_fields == 3
    assert list(ty) == fields
    assert ty[0].name == 'a'
    assert ty[2].type == pa.int32()
    with pytest.raises(IndexError):
        assert ty[3]
    assert ty['b'] == ty[2]
    assert ty['b'] == ty.field('b')
    assert ty[2] == ty.field(2)
    with pytest.raises(KeyError):
        ty['c']
    with pytest.raises(KeyError):
        ty.field('c')
    with pytest.raises(TypeError):
        ty[None]
    with pytest.raises(TypeError):
        ty.field(None)
    for a, b in zip(ty, fields):
        a == b
    ty = pa.struct([('a', pa.int64()), ('a', pa.int32()), ('b', pa.int32())])
    assert list(ty) == fields
    for a, b in zip(ty, fields):
        a == b
    fields = [pa.field('a', pa.int64()), pa.field('b', pa.int32())]
    ty = pa.struct(OrderedDict([('a', pa.int64()), ('b', pa.int32())]))
    assert list(ty) == fields
    for a, b in zip(ty, fields):
        a == b
    with pytest.raises(TypeError):
        pa.struct([('a', None)])