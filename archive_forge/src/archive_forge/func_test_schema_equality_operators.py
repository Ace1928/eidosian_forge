from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
def test_schema_equality_operators():
    fields = [pa.field('foo', pa.int32()), pa.field('bar', pa.string()), pa.field('baz', pa.list_(pa.int8()))]
    metadata = {b'foo': b'bar', b'pandas': b'badger'}
    sch1 = pa.schema(fields)
    sch2 = pa.schema(fields)
    sch3 = pa.schema(fields, metadata=metadata)
    sch4 = pa.schema(fields, metadata=metadata)
    assert sch1 == sch2
    assert sch3 == sch4
    assert sch1 == sch3
    assert not sch1 != sch3
    assert sch2 == sch4
    assert sch1 != []
    assert sch3 != 'foo'