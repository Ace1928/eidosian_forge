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
def batch_block_refs(block_refs: Iterator[ObjectRef[Block]], *, stats: Optional[DatasetStats]=None, prefetch_blocks: int=0, clear_block_after_read: bool=False, batch_size: Optional[int]=None, batch_format: str='default', drop_last: bool=False, collate_fn: Optional[Callable[[DataBatch], Any]]=None, shuffle_buffer_min_size: Optional[int]=None, shuffle_seed: Optional[int]=None, ensure_copy: bool=False) -> Iterator[DataBatch]:
    """Create formatted batches of data from 1 or more block object references.

    This takes a block iterator and creates batch_size batches, slicing,
    unioning, shuffling, prefetching, and formatting blocks as needed.

    This is used by both Dataset.iter_batches() and Dataset.map_batches().

    Args:
        block_refs: An iterator over block object references.
        prefetch_blocks: The number of blocks to prefetch ahead of the
            current block during the scan.
        clear_block_after_read: Whether to clear the block from object store
            manually (i.e. without waiting for Python's automatic GC) after it
            is read. Doing so will reclaim memory faster and hence reduce the
            memory footprint. However, the caller has to ensure the safety, i.e.
            the block will never be accessed again.
        batch_size: Record batch size, or None to let the system pick.
        batch_format: The format in which to return each batch.
            Specify "default" to use the current block format (promoting
            Arrow to pandas automatically), "pandas" to
            select ``pandas.DataFrame`` or "pyarrow" to select
            ``pyarrow.Table``. Default is "default".
        drop_last: Whether to drop the last batch if it's incomplete.
        collate_fn: A function to apply to each data batch before returning it.
        shuffle_buffer_min_size: If non-None, the data will be randomly shuffled using a
            local in-memory shuffle buffer, and this value will serve as the minimum
            number of rows that must be in the local in-memory shuffle buffer in order
            to yield a batch.
        shuffle_seed: The seed to use for the local random shuffle.
        ensure_copy: Whether batches are always copied from the underlying base
            blocks (not zero-copy views).

    Returns:
        An iterator over record batches.
    """
    context = DataContext.get_current()
    if prefetch_blocks > 0 and context.actor_prefetcher_enabled and (not ray.util.client.ray.is_connected()):
        prefetcher = ActorBlockPrefetcher()
    else:
        prefetcher = WaitBlockPrefetcher()
    eager_free = clear_block_after_read and DataContext.get_current().eager_free
    block_iter = resolve_block_refs(_prefetch_blocks(block_ref_iter=block_refs, prefetcher=prefetcher, num_blocks_to_prefetch=prefetch_blocks, eager_free=eager_free), stats=stats)
    yield from batch_blocks(block_iter, stats=stats, batch_size=batch_size, batch_format=batch_format, drop_last=drop_last, collate_fn=collate_fn, shuffle_buffer_min_size=shuffle_buffer_min_size, shuffle_seed=shuffle_seed, ensure_copy=ensure_copy)