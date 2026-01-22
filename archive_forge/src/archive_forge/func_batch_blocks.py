import collections
import itertools
from contextlib import nullcontext
from typing import Any, Callable, Iterator, Optional, TypeVar
import ray
from ray.data._internal.block_batching.interfaces import BlockPrefetcher
from ray.data._internal.block_batching.util import (
from ray.data._internal.memory_tracing import trace_deallocation
from ray.data._internal.stats import DatasetStats
from ray.data.block import Block, DataBatch
from ray.data.context import DataContext
from ray.types import ObjectRef
def batch_blocks(blocks: Iterator[Block], *, stats: Optional[DatasetStats]=None, batch_size: Optional[int]=None, batch_format: str='default', drop_last: bool=False, collate_fn: Optional[Callable[[DataBatch], DataBatch]]=None, shuffle_buffer_min_size: Optional[int]=None, shuffle_seed: Optional[int]=None, ensure_copy: bool=False) -> Iterator[DataBatch]:
    """Create formatted batches of data from 1 or more blocks.

    This is equivalent to batch_block_refs, except
    it takes in an iterator consisting of already fetched blocks.
    This means that this function does not support block prefetching.
    """

    def _iterator_fn(base_iterator: Iterator[Block]) -> Iterator[DataBatch]:
        batch_iter = format_batches(blocks_to_batches(block_iter=base_iterator, stats=stats, batch_size=batch_size, drop_last=drop_last, shuffle_buffer_min_size=shuffle_buffer_min_size, shuffle_seed=shuffle_seed, ensure_copy=ensure_copy), batch_format=batch_format, stats=stats)
        if collate_fn is not None:
            batch_iter = collate(batch_iter, collate_fn=collate_fn, stats=stats)
        batch_iter = extract_data_from_batch(batch_iter)
        yield from batch_iter
    batch_iter = _iterator_fn(blocks)
    for formatted_batch in batch_iter:
        user_timer = stats.iter_user_s.timer() if stats else nullcontext()
        with user_timer:
            yield formatted_batch