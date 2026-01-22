import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def par_iter_init(self, transforms):
    """Implements ParallelIterator worker init."""
    it = LocalIterator(lambda timeout: self.item_generator, SharedMetrics())
    for fn in transforms:
        it = fn(it)
        assert it is not None, fn
    self.local_it = iter(it)