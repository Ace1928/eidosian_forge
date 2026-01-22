from __future__ import annotations
import collections.abc
import inspect
import os
import signal
import threading
from abc import ABCMeta
from functools import update_wrapper
from typing import (
from sniffio import thread_local as sniffio_loop
import trio
def name_asyncgen(agen: AsyncGeneratorType[object, NoReturn]) -> str:
    """Return the fully-qualified name of the async generator function
    that produced the async generator iterator *agen*.
    """
    if not hasattr(agen, 'ag_code'):
        return repr(agen)
    try:
        module = agen.ag_frame.f_globals['__name__']
    except (AttributeError, KeyError):
        module = f'<{agen.ag_code.co_filename}>'
    try:
        qualname = agen.__qualname__
    except AttributeError:
        qualname = agen.ag_code.co_name
    return f'{module}.{qualname}'