import os
import sys
import threading
import weakref
from celery.local import Proxy
from celery.utils.threads import LocalStack
def get_current_worker_task():
    """Currently executing task, that was applied by the worker.

    This is used to differentiate between the actual task
    executed by the worker and any task that was called within
    a task (using ``task.__call__`` or ``task.apply``)
    """
    for task in reversed(_task_stack.stack):
        if not task.request.called_directly:
            return task