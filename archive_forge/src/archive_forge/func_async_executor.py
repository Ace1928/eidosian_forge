import asyncio
import collections
import contextvars
import datetime as dt
import inspect
import functools
import numbers
import os
import re
import sys
import traceback
import warnings
from collections import defaultdict, OrderedDict
from contextlib import contextmanager
from numbers import Real
from textwrap import dedent
from threading import get_ident
from collections import abc
def async_executor(func):
    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = asyncio.new_event_loop()
    if event_loop.is_running():
        task = asyncio.ensure_future(func())
        _running_tasks.add(task)
        task.add_done_callback(_running_tasks.discard)
    else:
        event_loop.run_until_complete(func())