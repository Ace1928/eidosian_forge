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
def mdayset(self, year, month, day):
    dset = [None] * self.yearlen
    start, end = self.mrange[month - 1:month + 1]
    for i in range(start, end):
        dset[i] = i
    return (dset, start, end)