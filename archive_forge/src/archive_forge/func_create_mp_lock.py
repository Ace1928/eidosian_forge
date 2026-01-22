import sys
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from numbers import Number
from time import time
from warnings import warn
from weakref import WeakSet
from ._monitor import TMonitor
from .utils import (
@classmethod
def create_mp_lock(cls):
    if not hasattr(cls, 'mp_lock'):
        try:
            from multiprocessing import RLock
            cls.mp_lock = RLock()
        except (ImportError, OSError):
            cls.mp_lock = None