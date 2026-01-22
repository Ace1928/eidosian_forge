import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def shards(self) -> List['LocalIterator[T]']:
    """Return the list of all shards."""
    return [self.get_shard(i) for i in range(self.num_shards())]