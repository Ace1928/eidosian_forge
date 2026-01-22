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
def set_postfix_str(self, s='', refresh=True):
    """
        Postfix without dictionary expansion, similar to prefix handling.
        """
    self.postfix = str(s)
    if refresh:
        self.refresh()