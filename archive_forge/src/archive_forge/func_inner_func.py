import calendar
import datetime
import heapq
import itertools
import re
import sys
from functools import wraps
from warnings import warn
from six import advance_iterator, integer_types
from six.moves import _thread, range
from ._common import weekday as weekdaybase
@wraps(f)
def inner_func(self, *args, **kwargs):
    rv = f(self, *args, **kwargs)
    self._invalidate_cache()
    return rv