import logging
import threading
from contextlib import nullcontext
from typing import Any, Callable, Iterator, List, Optional, Tuple
import ray
from ray.actor import ActorHandle
from ray.data._internal.batcher import Batcher, ShufflingBatcher
from ray.data._internal.block_batching.interfaces import (
from ray.data._internal.stats import DatasetStats
from ray.data.block import Block, BlockAccessor, DataBatch
from ray.types import ObjectRef
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def blocks_to_batches(block_iter: Iterator[Block], stats: Optional[DatasetStats]=None, batch_size: Optional[int]=None, drop_last: bool=False, shuffle_buffer_min_size: Optional[int]=None, shuffle_seed: Optional[int]=None, ensure_copy: bool=False) -> Iterator[Batch]:
    """Given an iterator over blocks, returns an iterator over blocks
    of the appropriate bacth size.

    If the shuffling configurations are specified, then the
    output blocks contain shuffled data.

    Args:
        block_iter: An iterator over blocks.
        stats: Dataset stats object used to store block batching time.
        batch_size: Record batch size, or None to let the system pick.
        drop_last: Whether to drop the last batch if it's incomplete.
        shuffle_buffer_min_size: If non-None, the data will be randomly shuffled
            using a local in-memory shuffle buffer, and this value will serve as the
            minimum number of rows that must be in the local in-memory shuffle buffer in
            order to yield a batch.
        shuffle_seed: The seed to use for the local random shuffle.
        ensure_copy: Whether batches are always copied from the underlying base
            blocks (not zero-copy views).

    Returns:
        An iterator over blocks of the given size that are potentially shuffled.
    """
    if shuffle_buffer_min_size is not None:
        batcher = ShufflingBatcher(batch_size=batch_size, shuffle_buffer_min_size=shuffle_buffer_min_size, shuffle_seed=shuffle_seed)
    else:
        batcher = Batcher(batch_size=batch_size, ensure_copy=ensure_copy)

    def get_iter_next_batch_s_timer():
        return stats.iter_next_batch_s.timer() if stats else nullcontext()
    global_counter = 0
    for block in block_iter:
        batcher.add(block)
        while batcher.has_batch():
            with get_iter_next_batch_s_timer():
                batch = batcher.next_batch()
            yield Batch(global_counter, batch)
            global_counter += 1
    batcher.done_adding()
    while batcher.has_batch():
        with get_iter_next_batch_s_timer():
            batch = batcher.next_batch()
        yield Batch(global_counter, batch)
        global_counter += 1
    if not drop_last and batcher.has_any():
        with get_iter_next_batch_s_timer():
            batch = batcher.next_batch()
        yield Batch(global_counter, batch)
        global_counter += 1