import fnmatch
import re
from collections import OrderedDict
from collections.abc import Mapping
from kombu import Queue
from celery.exceptions import QueueNotFound
from celery.utils.collections import lpmerge
from celery.utils.functional import maybe_evaluate, mlazy
from celery.utils.imports import symbol_by_name
def query_router(self, router, task, args, kwargs, options, task_type):
    router = maybe_evaluate(router)
    if hasattr(router, 'route_for_task'):
        return router.route_for_task(task, args, kwargs)
    return router(task, args, kwargs, options, task=task_type)