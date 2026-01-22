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
def ddayset(self, year, month, day):
    dset = [None] * self.yearlen
    i = datetime.date(year, month, day).toordinal() - self.yearordinal
    dset[i] = i
    return (dset, i, i + 1)