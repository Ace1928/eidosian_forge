import datetime
import pytest
import pyarrow as pa
@pytest.mark.gandiva
def test_filter_project():
    import pyarrow.gandiva as gandiva
    mpool = pa.default_memory_pool()
    array0 = pa.array([10, 12, -20, 5, 21, 29], pa.int32())
    array1 = pa.array([5, 15, 15, 17, 12, 3], pa.int32())
    array2 = pa.array([1, 25, 11, 30, -21, None], pa.int32())
    table = pa.Table.from_arrays([array0, array1, array2], ['a', 'b', 'c'])
    field_result = pa.field('res', pa.int32())
    builder = gandiva.TreeExprBuilder()
    node_a = builder.make_field(table.schema.field('a'))
    node_b = builder.make_field(table.schema.field('b'))
    node_c = builder.make_field(table.schema.field('c'))
    greater_than_function = builder.make_function('greater_than', [node_a, node_b], pa.bool_())
    filter_condition = builder.make_condition(greater_than_function)
    project_condition = builder.make_function('less_than', [node_b, node_c], pa.bool_())
    if_node = builder.make_if(project_condition, node_b, node_c, pa.int32())
    expr = builder.make_expression(if_node, field_result)
    filter = gandiva.make_filter(table.schema, filter_condition)
    projector = gandiva.make_projector(table.schema, [expr], mpool, 'UINT32')
    selection_vector = filter.evaluate(table.to_batches()[0], mpool)
    r, = projector.evaluate(table.to_batches()[0], selection_vector)
    exp = pa.array([1, -21, None], pa.int32())
    assert r.equals(exp)