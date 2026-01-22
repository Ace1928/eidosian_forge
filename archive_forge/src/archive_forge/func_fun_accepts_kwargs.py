import inspect
from collections import UserList
from functools import partial
from itertools import islice, tee, zip_longest
from typing import Any, Callable
from kombu.utils.functional import LRUCache, dictfilter, is_list, lazy, maybe_evaluate, maybe_list, memoize
from vine import promise
from celery.utils.log import get_logger
def fun_accepts_kwargs(fun):
    """Return true if function accepts arbitrary keyword arguments."""
    return any((p for p in inspect.signature(fun).parameters.values() if p.kind == p.VAR_KEYWORD))