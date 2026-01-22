import atexit
import threading
from collections import defaultdict
from collections import OrderedDict
from dataclasses import dataclass
from multiprocessing.pool import ThreadPool
from typing import Optional
import ray
import dask
from dask.core import istask, ishashable, _execute_task
from dask.system import CPU_COUNT
from dask.threaded import pack_exception, _thread_get_id
from ray.util.dask.callbacks import local_ray_callbacks, unpack_ray_callbacks
from ray.util.dask.common import unpack_object_refs
from ray.util.dask.scheduler_utils import get_async, apply_sync
def enable_dask_on_ray(shuffle: Optional[str]='tasks', use_shuffle_optimization: Optional[bool]=True) -> dask.config.set:
    """
    Enable Dask-on-Ray scheduler. This helper sets the Dask-on-Ray scheduler
    as the default Dask scheduler in the Dask config. By default, it will also
    cause the task-based shuffle to be used for any Dask shuffle operations
    (required for multi-node Ray clusters, not sharing a filesystem), and will
    enable a Ray-specific shuffle optimization.

    >>> enable_dask_on_ray()
    >>> ddf.compute()  # <-- will use the Dask-on-Ray scheduler.

    If used as a context manager, the Dask-on-Ray scheduler will only be used
    within the context's scope.

    >>> with enable_dask_on_ray():
    ...     ddf.compute()  # <-- will use the Dask-on-Ray scheduler.
    >>> ddf.compute()  # <-- won't use the Dask-on-Ray scheduler.

    Args:
        shuffle: The shuffle method used by Dask, either "tasks" or
            "disk". This should be "tasks" if using a multi-node Ray cluster.
            Defaults to "tasks".
        use_shuffle_optimization: Enable our custom Ray-specific shuffle
            optimization. Defaults to True.
    Returns:
        The Dask config object, which can be used as a context manager to limit
        the scope of the Dask-on-Ray scheduler to the corresponding context.
    """
    if use_shuffle_optimization:
        from ray.util.dask.optimizations import dataframe_optimize
    else:
        dataframe_optimize = None
    return dask.config.set(scheduler=ray_dask_get, shuffle=shuffle, dataframe_optimize=dataframe_optimize)