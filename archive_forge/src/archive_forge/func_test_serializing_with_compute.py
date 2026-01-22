import os
import pathlib
import pytest
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.lib import tobytes
from pyarrow.lib import ArrowInvalid, ArrowNotImplementedError
def test_serializing_with_compute():
    schema = pa.schema([pa.field('x', pa.int32()), pa.field('y', pa.int32())])
    expr = pc.equal(pc.field('x'), 7)
    expr_norm = pc.equal(pc.field(0), 7)
    buf = expr.to_substrait(schema)
    returned = pa.substrait.deserialize_expressions(buf)
    assert schema == returned.schema
    assert len(returned.expressions) == 1
    assert str(returned.expressions['expression']) == str(expr_norm)
    buf = pa.substrait.serialize_expressions([expr, expr], ['first', 'second'], schema)
    with pytest.raises(ValueError) as excinfo:
        pc.Expression.from_substrait(buf)
    assert 'contained multiple expressions' in str(excinfo.value)
    buf = pa.substrait.serialize_expressions([expr], ['weirdname'], schema)
    expr2 = pc.Expression.from_substrait(buf)
    assert str(expr2) == str(expr_norm)