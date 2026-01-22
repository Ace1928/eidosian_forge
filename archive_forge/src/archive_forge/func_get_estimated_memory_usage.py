import collections
from typing import Any, Mapping
from ray.data._internal.arrow_block import ArrowBlockBuilder
from ray.data._internal.block_builder import BlockBuilder
from ray.data._internal.pandas_block import PandasBlockBuilder
from ray.data.block import Block, BlockAccessor, DataBatch
def get_estimated_memory_usage(self) -> int:
    if self._builder is None:
        return 0
    return self._builder.get_estimated_memory_usage()