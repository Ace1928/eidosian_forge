import random
import threading
from oslo_utils import reflection
from oslo_utils import timeutils
from taskflow.engines.worker_based import protocol as pr
from taskflow import logging
from taskflow.utils import kombu_utils as ku
def get_worker_for_task(self, task):
    """Gets a worker that can perform a given task."""
    available_workers = []
    with self._cond:
        for worker in self._workers.values():
            if worker.performs(task):
                available_workers.append(worker)
    if available_workers:
        return self._match_worker(task, available_workers)
    else:
        return None