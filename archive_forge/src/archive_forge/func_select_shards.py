import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def select_shards(self, shards_to_keep: List[int]) -> 'ParallelIterator[T]':
    """Return a child iterator that only iterates over given shards.

        It is the user's responsibility to ensure child iterators are operating
        over disjoint sub-sets of this iterator's shards.
        """
    if len(self.actor_sets) > 1:
        raise ValueError('select_shards() is not allowed after union()')
    if len(shards_to_keep) == 0:
        raise ValueError('at least one shard must be selected')
    old_actor_set = self.actor_sets[0]
    new_actors = [a for i, a in enumerate(old_actor_set.actors) if i in shards_to_keep]
    assert len(new_actors) == len(shards_to_keep), 'Invalid actor index'
    new_actor_set = _ActorSet(new_actors, old_actor_set.transforms)
    return ParallelIterator([new_actor_set], f'{self}.select_shards({len(shards_to_keep)} total)', parent_iterators=self.parent_iterators)