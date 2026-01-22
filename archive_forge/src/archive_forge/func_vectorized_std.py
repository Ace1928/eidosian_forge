import math
from typing import TYPE_CHECKING, Callable, List, Optional, Union
from ray.data._internal.null_aggregate import (
from ray.data._internal.sort import SortKey
from ray.data.block import AggType, Block, BlockAccessor, KeyType, T, U
from ray.util.annotations import PublicAPI
def vectorized_std(block: Block) -> AggType:
    block_acc = BlockAccessor.for_block(block)
    count = block_acc.count(on)
    if count == 0 or count is None:
        return None
    sum_ = block_acc.sum(on, ignore_nulls)
    if sum_ is None:
        return None
    mean = sum_ / count
    M2 = block_acc.sum_of_squared_diffs_from_mean(on, ignore_nulls, mean)
    return [M2, mean, count]