from __future__ import annotations
import os
import signal
import contextlib
import multiprocessing
from typing import Optional, List, TypeVar, Callable, Dict, Any, Union, TYPE_CHECKING
from lazyops.utils.lazy import lazy_import
from lazyops.imports._psutil import _psutil_available
def has_worker_processes(cls, name: str, kind: Optional[str]=None) -> bool:
    """
        Checks if there are processes
        """
    if kind is None:
        kind = 'default'
    if kind not in cls.worker_processes:
        cls.worker_processes[kind] = {}
    if name not in cls.worker_processes[kind]:
        cls.worker_processes[kind][name] = []
    return len(cls.worker_processes[kind][name]) > 0