import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def par_iter_next(self):
    """Implements ParallelIterator worker item fetch."""
    assert self.local_it is not None, 'must call par_iter_init()'
    return next(self.local_it)