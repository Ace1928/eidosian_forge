import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def cycle():
    while True:
        it = iter(make_iterator())
        if it is item_generator:
            raise ValueError('Cannot iterate over {0} multiple times.' + 'Please pass in the base iterable or' + 'lambda: {0} instead.'.format(item_generator))
        for item in it:
            yield item