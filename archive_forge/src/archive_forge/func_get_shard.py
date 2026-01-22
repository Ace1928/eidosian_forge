import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def get_shard(self, shard_index: int, batch_ms: int=0, num_async: int=1) -> 'LocalIterator[T]':
    """Return a local iterator for the given shard.

        The iterator is guaranteed to be serializable and can be passed to
        remote tasks or actors.

        Arguments:
            shard_index: Index of the shard to gather.
            batch_ms: Batches items for batch_ms milliseconds
                before retrieving it.
                Increasing batch_ms increases latency but improves throughput.
                If this value is 0, then items are returned immediately.
            num_async: The max number of requests in flight.
                Increasing this improves the amount of pipeline
                parallelism in the iterator.
        """
    if num_async < 1:
        raise ValueError('num async must be positive')
    if batch_ms < 0:
        raise ValueError('batch time must be positive')
    a, t = (None, None)
    i = shard_index
    for actor_set in self.actor_sets:
        if i < len(actor_set.actors):
            a = actor_set.actors[i]
            t = actor_set.transforms
            break
        else:
            i -= len(actor_set.actors)
    if a is None:
        raise ValueError('Shard index out of range', shard_index, self.num_shards())

    def base_iterator(timeout=None):
        queue = collections.deque()
        ray.get(a.par_iter_init.remote(t))
        for _ in range(num_async):
            queue.append(a.par_iter_next_batch.remote(batch_ms))
        while True:
            try:
                batch = ray.get(queue.popleft(), timeout=timeout)
                queue.append(a.par_iter_next_batch.remote(batch_ms))
                for item in batch:
                    yield item
                if timeout is not None:
                    yield _NextValueNotReady()
            except TimeoutError:
                yield _NextValueNotReady()
            except StopIteration:
                break
    name = self.name + f'.shard[{shard_index}]'
    return LocalIterator(base_iterator, SharedMetrics(), name=name)