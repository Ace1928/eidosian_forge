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
def ray_get_unpack(object_refs, progress_bar_actor=None):
    """
    Unpacks object references, gets the object references, and repacks.
    Traverses arbitrary data structures.

    Args:
        object_refs: A (potentially nested) Python object containing Ray object
            references.

    Returns:
        The input Python object with all contained Ray object references
        resolved with their concrete values.
    """

    def get_result(object_refs):
        if progress_bar_actor:
            render_progress_bar(progress_bar_actor, object_refs)
        return ray.get(object_refs)
    if isinstance(object_refs, tuple):
        object_refs = list(object_refs)
    if isinstance(object_refs, list) and any((not isinstance(x, ray.ObjectRef) for x in object_refs)):
        object_refs, repack = unpack_object_refs(*object_refs)
        computed_result = get_result(object_refs)
        return repack(computed_result)
    else:
        return get_result(object_refs)