import datetime
import pytest
import pyarrow as pa
@pytest.mark.gandiva
def test_rejects_none():
    import pyarrow.gandiva as gandiva
    builder = gandiva.TreeExprBuilder()
    field_x = pa.field('x', pa.int32())
    schema = pa.schema([field_x])
    literal_true = builder.make_literal(True, pa.bool_())
    with pytest.raises(TypeError):
        builder.make_field(None)
    with pytest.raises(TypeError):
        builder.make_if(literal_true, None, None, None)
    with pytest.raises(TypeError):
        builder.make_and([literal_true, None])
    with pytest.raises(TypeError):
        builder.make_or([None, literal_true])
    with pytest.raises(TypeError):
        builder.make_in_expression(None, [1, 2, 3], pa.int32())
    with pytest.raises(TypeError):
        builder.make_expression(None, field_x)
    with pytest.raises(TypeError):
        builder.make_condition(None)
    with pytest.raises(TypeError):
        builder.make_function('less_than', [literal_true, None], pa.bool_())
    with pytest.raises(TypeError):
        gandiva.make_projector(schema, [None])
    with pytest.raises(TypeError):
        gandiva.make_filter(schema, None)