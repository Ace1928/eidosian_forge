import os
import pathlib
import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import tobytes
from pyarrow.lib import ArrowInvalid, ArrowNotImplementedError
@pytest.mark.parametrize('expr', [pc.equal(pc.field('x'), 7), pc.equal(pc.field('x'), pc.field('y')), pc.field('x') > 50])
def test_serializing_expressions(expr):
    schema = pa.schema([pa.field('x', pa.int32()), pa.field('y', pa.int32())])
    buf = pa.substrait.serialize_expressions([expr], ['test_expr'], schema)
    returned = pa.substrait.deserialize_expressions(buf)
    assert schema == returned.schema
    assert len(returned.expressions) == 1
    assert 'test_expr' in returned.expressions