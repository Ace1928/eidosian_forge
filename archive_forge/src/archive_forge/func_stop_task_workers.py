from __future__ import annotations
import os
import abc
import signal
import contextlib
import multiprocessing
import pathlib
import asyncio
from typing import Optional, List, TypeVar, Callable, Dict, Any, overload, Type, Union, TYPE_CHECKING
from lazyops.utils.lazy import lazy_import
from lazyops.libs.proxyobj import ProxyObject, proxied
from lazyops.imports._psutil import _psutil_available
def stop_task_workers(self, worker_names: Optional[List[str]]=None, timeout: Optional[float]=5.0, verbose: bool=True, **kwargs):
    """
        Stops the task workers
        """
    if worker_names is None:
        worker_names = list(self.workers.keys())
    from kvdb.tasks.spawn import exit_task_workers
    exit_task_workers(worker_names=worker_names, timeout=timeout)
    for worker_name in worker_names:
        if verbose:
            self.logger.info(f'Stopped worker: {worker_name}')
        _ = self.workers.pop(worker_name, None)