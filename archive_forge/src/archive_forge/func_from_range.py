import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
@Deprecated
def from_range(n: int, num_shards: int=2, repeat: bool=False) -> 'ParallelIterator[int]':
    """Create a parallel iterator over the range 0..n.

    The range will be partitioned sequentially among the number of shards.

    Args:
        n: The max end of the range of numbers.
        num_shards: The number of worker actors to create.
        repeat: Whether to cycle over the range forever.
    """
    generators = []
    shard_size = n // num_shards
    for i in range(num_shards):
        start = i * shard_size
        if i == num_shards - 1:
            end = n
        else:
            end = (i + 1) * shard_size
        generators.append(range(start, end))
    name = f'from_range[{n}, shards={num_shards}{(', repeat=True' if repeat else '')}]'
    return from_iterators(generators, repeat=repeat, name=name)