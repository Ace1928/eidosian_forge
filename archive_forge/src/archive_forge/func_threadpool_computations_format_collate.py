import collections
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterator, Optional, Tuple
import ray
from ray.data._internal.block_batching.interfaces import Batch, BlockPrefetcher
from ray.data._internal.block_batching.util import (
from ray.data._internal.memory_tracing import trace_deallocation
from ray.data._internal.stats import DatasetStats
from ray.data._internal.util import make_async_gen
from ray.data.block import Block, BlockMetadata, DataBatch
from ray.data.context import DataContext
from ray.types import ObjectRef
def threadpool_computations_format_collate(batch_iter: Iterator[Batch]) -> Iterator[Batch]:
    formatted_batch_iter = format_batches(batch_iter, batch_format=batch_format, stats=stats)
    if collate_fn is not None:
        formatted_batch_iter = collate(formatted_batch_iter, collate_fn=collate_fn, stats=stats)
    yield from formatted_batch_iter