import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def zip_with_source(item):
    metrics = LocalIterator.get_metrics()
    if metrics.current_actor is None:
        raise ValueError('Could not identify source actor of item')
    return (metrics.current_actor, item)