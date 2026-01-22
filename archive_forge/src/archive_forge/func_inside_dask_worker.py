import gc
import os
import warnings
import threading
import contextlib
from abc import ABCMeta, abstractmethod
from ._multiprocessing_helpers import mp
def inside_dask_worker():
    """Check whether the current function is executed inside a Dask worker.
    """
    try:
        from distributed import get_worker
    except ImportError:
        return False
    try:
        get_worker()
        return True
    except ValueError:
        return False