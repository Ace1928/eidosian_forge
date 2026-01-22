import datetime
import pytest
import pyarrow as pa
@pytest.mark.gandiva
def test_in_expr():
    import pyarrow.gandiva as gandiva
    arr = pa.array(['ga', 'an', 'nd', 'di', 'iv', 'va'])
    table = pa.Table.from_arrays([arr], ['a'])
    builder = gandiva.TreeExprBuilder()
    node_a = builder.make_field(table.schema.field('a'))
    cond = builder.make_in_expression(node_a, ['an', 'nd'], pa.string())
    condition = builder.make_condition(cond)
    filter = gandiva.make_filter(table.schema, condition)
    result = filter.evaluate(table.to_batches()[0], pa.default_memory_pool())
    assert result.to_array().equals(pa.array([1, 2], type=pa.uint32()))
    arr = pa.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 4])
    table = pa.Table.from_arrays([arr.cast(pa.int32())], ['a'])
    node_a = builder.make_field(table.schema.field('a'))
    cond = builder.make_in_expression(node_a, [1, 5], pa.int32())
    condition = builder.make_condition(cond)
    filter = gandiva.make_filter(table.schema, condition)
    result = filter.evaluate(table.to_batches()[0], pa.default_memory_pool())
    assert result.to_array().equals(pa.array([1, 3, 4, 8], type=pa.uint32()))
    arr = pa.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 4])
    table = pa.Table.from_arrays([arr], ['a'])
    node_a = builder.make_field(table.schema.field('a'))
    cond = builder.make_in_expression(node_a, [1, 5], pa.int64())
    condition = builder.make_condition(cond)
    filter = gandiva.make_filter(table.schema, condition)
    result = filter.evaluate(table.to_batches()[0], pa.default_memory_pool())
    assert result.to_array().equals(pa.array([1, 3, 4, 8], type=pa.uint32()))