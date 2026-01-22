from typing import Any
from ray.data._internal.delegating_block_builder import DelegatingBlockBuilder
from ray.data.block import Block, BlockAccessor, DataBatch
from ray.data.context import MAX_SAFE_BLOCK_SIZE_FACTOR
Returns the next complete output block.