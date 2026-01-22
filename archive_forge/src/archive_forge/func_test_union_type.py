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
def test_union_type():

    def check_fields(ty, fields):
        assert ty.num_fields == len(fields)
        assert [ty[i] for i in range(ty.num_fields)] == fields
        assert [ty.field(i) for i in range(ty.num_fields)] == fields
    fields = [pa.field('x', pa.list_(pa.int32())), pa.field('y', pa.binary())]
    type_codes = [5, 9]
    sparse_factories = [partial(pa.union, mode='sparse'), partial(pa.union, mode=pa.lib.UnionMode_SPARSE), pa.sparse_union]
    dense_factories = [partial(pa.union, mode='dense'), partial(pa.union, mode=pa.lib.UnionMode_DENSE), pa.dense_union]
    for factory in sparse_factories:
        ty = factory(fields)
        assert isinstance(ty, pa.SparseUnionType)
        assert ty.mode == 'sparse'
        check_fields(ty, fields)
        assert ty.type_codes == [0, 1]
        ty = factory(fields, type_codes=type_codes)
        assert ty.mode == 'sparse'
        check_fields(ty, fields)
        assert ty.type_codes == type_codes
        with pytest.raises(ValueError):
            factory(fields, type_codes=type_codes[1:])
    for factory in dense_factories:
        ty = factory(fields)
        assert isinstance(ty, pa.DenseUnionType)
        assert ty.mode == 'dense'
        check_fields(ty, fields)
        assert ty.type_codes == [0, 1]
        ty = factory(fields, type_codes=type_codes)
        assert ty.mode == 'dense'
        check_fields(ty, fields)
        assert ty.type_codes == type_codes
        with pytest.raises(ValueError):
            factory(fields, type_codes=type_codes[1:])
    for mode in ('unknown', 2):
        with pytest.raises(ValueError, match='Invalid union mode'):
            pa.union(fields, mode=mode)