import inspect
from collections import UserList
from functools import partial
from itertools import islice, tee, zip_longest
from typing import Any, Callable
from kombu.utils.functional import LRUCache, dictfilter, is_list, lazy, maybe_evaluate, maybe_list, memoize
from vine import promise
from celery.utils.log import get_logger
def seq_concat_item(seq, item):
    """Return copy of sequence seq with item added.

    Returns:
        Sequence: if seq is a tuple, the result will be a tuple,
           otherwise it depends on the implementation of ``__add__``.
    """
    return seq + (item,) if isinstance(seq, tuple) else seq + [item]