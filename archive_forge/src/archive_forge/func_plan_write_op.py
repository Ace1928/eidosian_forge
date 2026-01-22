from typing import Callable, Iterator, Union
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
from ray.data._internal.logical.operators.write_operator import Write
from ray.data.block import Block
from ray.data.datasource.datasink import Datasink
from ray.data.datasource.datasource import Datasource
def plan_write_op(op: Write, input_physical_dag: PhysicalOperator) -> PhysicalOperator:
    write_fn = generate_write_fn(op._datasink_or_legacy_datasource, **op._write_args)
    transform_fns = [BlockMapTransformFn(write_fn)]
    map_transformer = MapTransformer(transform_fns)
    return MapOperator.create(map_transformer, input_physical_dag, name='Write', target_max_block_size=None, ray_remote_args=op._ray_remote_args)