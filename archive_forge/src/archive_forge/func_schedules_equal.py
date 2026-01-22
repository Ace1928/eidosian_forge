import copy
import errno
import heapq
import os
import shelve
import sys
import time
import traceback
from calendar import timegm
from collections import namedtuple
from functools import total_ordering
from threading import Event, Thread
from billiard import ensure_multiprocessing
from billiard.common import reset_signals
from billiard.context import Process
from kombu.utils.functional import maybe_evaluate, reprcall
from kombu.utils.objects import cached_property
from . import __version__, platforms, signals
from .exceptions import reraise
from .schedules import crontab, maybe_schedule
from .utils.functional import is_numeric_value
from .utils.imports import load_extension_class_names, symbol_by_name
from .utils.log import get_logger, iter_open_logger_fds
from .utils.time import humanize_seconds, maybe_make_aware
def schedules_equal(self, old_schedules, new_schedules):
    if old_schedules is new_schedules is None:
        return True
    if old_schedules is None or new_schedules is None:
        return False
    if set(old_schedules.keys()) != set(new_schedules.keys()):
        return False
    for name, old_entry in old_schedules.items():
        new_entry = new_schedules.get(name)
        if not new_entry:
            return False
        if new_entry != old_entry:
            return False
    return True