import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def par_iter_slice_batch(self, step: int, start: int, batch_ms: int):
    """Batches par_iter_slice."""
    batch = []
    if batch_ms == 0:
        batch.append(self.par_iter_slice(step, start))
        return batch
    t_end = time.time() + 0.001 * batch_ms
    while time.time() < t_end:
        try:
            batch.append(self.par_iter_slice(step, start))
        except StopIteration:
            if len(batch) == 0:
                raise StopIteration
            else:
                pass
    return batch