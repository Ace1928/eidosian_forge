from __future__ import division
import os
import sys
from math import sqrt
import functools
import collections
import time
import threading
import itertools
from uuid import uuid4
from numbers import Integral
import warnings
import queue
import weakref
from contextlib import nullcontext
from multiprocessing import TimeoutError
from ._multiprocessing_helpers import mp
from .logger import Logger, short_format_time
from .disk import memstr_to_bytes
from ._parallel_backends import (FallbackToBackend, MultiprocessingBackend,
from ._utils import eval_expr, _Sentinel
from ._parallel_backends import AutoBatchingMixin  # noqa
from ._parallel_backends import ParallelBackendBase  # noqa
def dispatch_one_batch(self, iterator):
    """Prefetch the tasks for the next batch and dispatch them.

        The effective size of the batch is computed here.
        If there are no more jobs to dispatch, return False, else return True.

        The iterator consumption and dispatching is protected by the same
        lock so calling this function should be thread safe.

        """
    if self._aborting:
        return False
    batch_size = self._get_batch_size()
    with self._lock:
        try:
            tasks = self._ready_batches.get(block=False)
        except queue.Empty:
            n_jobs = self._cached_effective_n_jobs
            big_batch_size = batch_size * n_jobs
            islice = list(itertools.islice(iterator, big_batch_size))
            if len(islice) == 0:
                return False
            elif iterator is self._original_iterator and len(islice) < big_batch_size:
                final_batch_size = max(1, len(islice) // (10 * n_jobs))
            else:
                final_batch_size = max(1, len(islice) // n_jobs)
            for i in range(0, len(islice), final_batch_size):
                tasks = BatchedCalls(islice[i:i + final_batch_size], self._backend.get_nested_backend(), self._reducer_callback, self._pickle_cache)
                self._ready_batches.put(tasks)
            tasks = self._ready_batches.get(block=False)
        if len(tasks) == 0:
            return False
        else:
            self._dispatch(tasks)
            return True