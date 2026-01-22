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
def render_progress_bar(tracker, object_refs):
    from tqdm import tqdm
    total, finished = ray.get(tracker.result.remote())
    reported_finished_so_far = 0
    pb_bar = tqdm(total=total, position=0)
    pb_bar.set_description('')
    ready_refs = []
    while finished < total:
        submitted, finished = ray.get(tracker.result.remote())
        pb_bar.update(finished - reported_finished_so_far)
        reported_finished_so_far = finished
        ready_refs, _ = ray.wait(object_refs, timeout=0, num_returns=len(object_refs), fetch_local=False)
        if len(ready_refs) == len(object_refs):
            break
        import time
        time.sleep(0.1)
    pb_bar.close()
    submitted, finished = ray.get(tracker.result.remote())
    if submitted != finished:
        print('Completed. There was state inconsistency.')
    from pprint import pprint
    pprint(ray.get(tracker.report.remote()))