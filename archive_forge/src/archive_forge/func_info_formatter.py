from __future__ import annotations
import atexit
import contextlib
import functools
import inspect
import itertools
import os
import pprint
import re
import reprlib
import sys
import traceback
import types
import _thread
from typing import (
from coverage.misc import human_sorted_items, isolate_module
from coverage.types import AnyCallable, TWritable
def info_formatter(info: Iterable[tuple[str, Any]]) -> Iterator[str]:
    """Produce a sequence of formatted lines from info.

    `info` is a sequence of pairs (label, data).  The produced lines are
    nicely formatted, ready to print.

    """
    info = list(info)
    if not info:
        return
    label_len = 30
    assert all((len(l) < label_len for l, _ in info))
    for label, data in info:
        if data == []:
            data = '-none-'
        if isinstance(data, tuple) and len(repr(tuple(data))) < 30:
            yield ('%*s: %r' % (label_len, label, tuple(data)))
        elif isinstance(data, (list, set, tuple)):
            prefix = '%*s:' % (label_len, label)
            for e in data:
                yield ('%*s %s' % (label_len + 1, prefix, e))
                prefix = ''
        else:
            yield ('%*s: %s' % (label_len, label, data))