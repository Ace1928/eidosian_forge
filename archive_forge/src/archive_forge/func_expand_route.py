import fnmatch
import re
from collections import OrderedDict
from collections.abc import Mapping
from kombu import Queue
from celery.exceptions import QueueNotFound
from celery.utils.collections import lpmerge
from celery.utils.functional import maybe_evaluate, mlazy
from celery.utils.imports import symbol_by_name
def expand_route(route):
    if isinstance(route, (Mapping, list, tuple)):
        return MapRoute(route)
    if isinstance(route, str):
        return mlazy(expand_router_string, route)
    return route