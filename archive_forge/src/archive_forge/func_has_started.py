import calendar
import datetime
import functools
import logging
import time
import iso8601
from oslo_utils import reflection
def has_started(self):
    """Returns True if the watch is in a started state."""
    return self._state == self._STARTED