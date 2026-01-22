from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
def test_field_flatten():
    f0 = pa.field('foo', pa.int32()).with_metadata({b'foo': b'bar'})
    assert f0.flatten() == [f0]
    f1 = pa.field('bar', pa.float64(), nullable=False)
    ff = pa.field('ff', pa.struct([f0, f1]), nullable=False)
    assert ff.flatten() == [pa.field('ff.foo', pa.int32()).with_metadata({b'foo': b'bar'}), pa.field('ff.bar', pa.float64(), nullable=False)]
    ff = pa.field('ff', pa.struct([f0, f1]))
    assert ff.flatten() == [pa.field('ff.foo', pa.int32()).with_metadata({b'foo': b'bar'}), pa.field('ff.bar', pa.float64())]
    fff = pa.field('fff', pa.struct([ff]))
    assert fff.flatten() == [pa.field('fff.ff', pa.struct([f0, f1]))]