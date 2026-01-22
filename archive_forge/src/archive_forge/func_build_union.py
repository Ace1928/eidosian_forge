import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def build_union(timeout=None):
    while True:
        for weight, it in list(active):
            if weight == '*':
                max_pull = 100
            else:
                max_pull = _randomized_int_cast(weight)
            try:
                for _ in range(max_pull):
                    item = next(it)
                    if isinstance(item, _NextValueNotReady):
                        if timeout is not None:
                            yield item
                        break
                    else:
                        yield item
            except StopIteration:
                active.remove((weight, it))
        if not active:
            break