import inspect
from collections import UserList
from functools import partial
from itertools import islice, tee, zip_longest
from typing import Any, Callable
from kombu.utils.functional import LRUCache, dictfilter, is_list, lazy, maybe_evaluate, maybe_list, memoize
from vine import promise
from celery.utils.log import get_logger
def fun_takes_argument(name, fun, position=None):
    spec = inspect.getfullargspec(fun)
    return spec.varkw or spec.varargs or (len(spec.args) >= position if position else name in spec.args)