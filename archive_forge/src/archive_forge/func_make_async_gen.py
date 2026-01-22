import importlib
import logging
import os
import pathlib
import random
import sys
import threading
import time
import urllib.parse
from collections import deque
from types import ModuleType
from typing import (
import numpy as np
import ray
from ray._private.utils import _get_pyarrow_version
from ray.data._internal.arrow_ops.transform_pyarrow import unify_schemas
from ray.data.context import WARN_PREFIX, DataContext
def make_async_gen(base_iterator: Iterator[T], fn: Callable[[Iterator[T]], Iterator[U]], num_workers: int=1) -> Iterator[U]:
    """Returns a new iterator with elements fetched from the base_iterator
    in an async fashion using a threadpool.

    Each thread in the threadpool will fetch data from the base_iterator in a
    thread-safe fashion, and apply the provided `fn` computation concurrently.

    Args:
        base_iterator: The iterator to asynchronously fetch from.
        fn: The function to run on the input iterator.
        num_workers: The number of threads to use in the threadpool. Defaults to 1.

    Returns:
        An iterator with the same elements as outputted from `fn`.
    """
    if num_workers < 1:
        raise ValueError('Size of threadpool must be at least 1.')

    def convert_to_threadsafe_iterator(base_iterator: Iterator[T]) -> Iterator[T]:

        class ThreadSafeIterator:

            def __init__(self, it):
                self.lock = threading.Lock()
                self.it = it

            def __next__(self):
                with self.lock:
                    return next(self.it)

            def __iter__(self):
                return self
        return ThreadSafeIterator(base_iterator)
    thread_safe_generator = convert_to_threadsafe_iterator(base_iterator)

    class Sentinel:

        def __init__(self, thread_index: int):
            self.thread_index = thread_index
    output_queue = Queue(1)

    def execute_computation(thread_index: int):
        try:
            for item in fn(thread_safe_generator):
                if output_queue.put(item):
                    return
            output_queue.put(Sentinel(thread_index))
        except Exception as e:
            output_queue.put(e)
    threads = [threading.Thread(target=execute_computation, args=(i,), daemon=True) for i in range(num_workers)]
    for thread in threads:
        thread.start()
    num_threads_finished = 0
    try:
        while True:
            next_item = output_queue.get()
            if isinstance(next_item, Exception):
                raise next_item
            if isinstance(next_item, Sentinel):
                logger.debug(f'Thread {next_item.thread_index} finished.')
                num_threads_finished += 1
            else:
                yield next_item
            if num_threads_finished >= num_workers:
                break
    finally:
        num_threads_alive = num_workers - num_threads_finished
        if num_threads_alive > 0:
            output_queue.release(num_threads_alive)