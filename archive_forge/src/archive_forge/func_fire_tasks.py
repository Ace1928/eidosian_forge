from __future__ import annotations
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor, Future
from functools import partial
from queue import Empty, Queue
from dask import config
from dask.callbacks import local_callbacks, unpack_callbacks
from dask.core import _execute_task, flatten, get_dependencies, has_tasks, reverse_dict
from dask.order import order
from dask.typing import Key
def fire_tasks(chunksize):
    """Fire off a task to the thread pool"""
    nready = len(state['ready'])
    if chunksize == -1:
        ntasks = nready
        chunksize = -(ntasks // -num_workers)
    else:
        used_workers = -(len(state['running']) // -chunksize)
        avail_workers = max(num_workers - used_workers, 0)
        ntasks = min(nready, chunksize * avail_workers)
    args = []
    for _ in range(ntasks):
        key = state['ready'].pop()
        state['running'].add(key)
        for f in pretask_cbs:
            f(key, dsk, state)
        data = {dep: state['cache'][dep] for dep in get_dependencies(dsk, key)}
        args.append((key, dumps((dsk[key], data)), dumps, loads, get_id, pack_exception))
    for i in range(-(len(args) // -chunksize)):
        each_args = args[i * chunksize:(i + 1) * chunksize]
        if not each_args:
            break
        fut = submit(batch_execute_tasks, each_args)
        fut.add_done_callback(queue.put)