import os
import pathlib
import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import tobytes
from pyarrow.lib import ArrowInvalid, ArrowNotImplementedError
def test_serializing_udfs():
    schema = pa.schema([pa.field('x', pa.uint32())])
    a = pc.scalar(10)
    b = pc.scalar(4)
    exprs = [pc.shift_left(a, b)]
    with pytest.raises(ArrowNotImplementedError):
        pa.substrait.serialize_expressions(exprs, ['expr'], schema)
    buf = pa.substrait.serialize_expressions(exprs, ['expr'], schema, allow_arrow_extensions=True)
    returned = pa.substrait.deserialize_expressions(buf)
    assert schema == returned.schema
    assert len(returned.expressions) == 1
    assert str(returned.expressions['expr']) == str(exprs[0])