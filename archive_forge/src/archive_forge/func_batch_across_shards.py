import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def batch_across_shards(self) -> 'LocalIterator[List[T]]':
    """Iterate over the results of multiple shards in parallel.

        Examples:
            >>> it = from_iterators([range(3), range(3)])
            >>> next(it.batch_across_shards())
            ... [0, 0]
        """

    def base_iterator(timeout=None):
        active = []
        for actor_set in self.actor_sets:
            actor_set.init_actors()
            active.extend(actor_set.actors)
        futures = [a.par_iter_next.remote() for a in active]
        while active:
            try:
                yield ray.get(futures, timeout=timeout)
                futures = [a.par_iter_next.remote() for a in active]
                if timeout is not None:
                    yield _NextValueNotReady()
            except TimeoutError:
                yield _NextValueNotReady()
            except StopIteration:
                results = []
                for a, f in zip(list(active), futures):
                    try:
                        results.append(ray.get(f))
                    except StopIteration:
                        active.remove(a)
                if results:
                    yield results
                futures = [a.par_iter_next.remote() for a in active]
    name = f'{self}.batch_across_shards()'
    return LocalIterator(base_iterator, SharedMetrics(), name=name)