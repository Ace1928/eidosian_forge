import fnmatch
import re
from collections import OrderedDict
from collections.abc import Mapping
from kombu import Queue
from celery.exceptions import QueueNotFound
from celery.utils.collections import lpmerge
from celery.utils.functional import maybe_evaluate, mlazy
from celery.utils.imports import symbol_by_name
def lookup_route(self, name, args=None, kwargs=None, options=None, task_type=None):
    query = self.query_router
    for router in self.routes:
        route = query(router, name, args, kwargs, options, task_type)
        if route is not None:
            return route