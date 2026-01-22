from __future__ import annotations
import asyncio
import datetime as dt
import inspect
import logging
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial, wraps
from typing import (
from urllib.parse import urljoin
from weakref import WeakKeyDictionary
import param
from bokeh.document import Document
from bokeh.document.locking import UnlockedDocumentProxy
from bokeh.io import curdoc as _curdoc
from pyviz_comms import CommManager as _CommManager
from ..util import decode_token, parse_timedelta
from .logging import LOG_SESSION_RENDERED, LOG_USER_MSG
def cancel_task(self, name: str, wait: bool=False):
    """
        Cancel a task scheduled using the `state.schedule_task` method by name.

        Arguments
        ---------
        name: str
            The name of the scheduled task.
        wait: boolean
            Whether to wait until after the next execution.
        """
    if name not in self._scheduled:
        raise KeyError(f'No task with the name {name!r} has been scheduled.')
    if wait:
        self._scheduled[name] = (None, self._scheduled[name][1])
    else:
        del self._scheduled[name]