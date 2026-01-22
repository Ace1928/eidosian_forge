from typing import Callable, Iterator, Union
from ray.data._internal.execution.interfaces import PhysicalOperator
from ray.data._internal.execution.interfaces.task_context import TaskContext
from ray.data._internal.execution.operators.map_operator import MapOperator
from ray.data._internal.execution.operators.map_transformer import (
from ray.data._internal.logical.operators.write_operator import Write
from ray.data.block import Block
from ray.data.datasource.datasink import Datasink
from ray.data.datasource.datasource import Datasource
def generate_write_fn(datasink_or_legacy_datasource: Union[Datasink, Datasource], **write_args) -> Callable[[Iterator[Block], TaskContext], Iterator[Block]]:

    def fn(blocks: Iterator[Block], ctx) -> Iterator[Block]:
        if isinstance(datasink_or_legacy_datasource, Datasink):
            write_result = datasink_or_legacy_datasource.write(blocks, ctx)
        else:
            write_result = datasink_or_legacy_datasource.write(blocks, ctx, **write_args)
        import pandas as pd
        block = pd.DataFrame({'write_result': [write_result]})
        return [block]
    return fn