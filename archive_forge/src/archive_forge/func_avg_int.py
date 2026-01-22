from fontTools.misc.timeTools import timestampNow
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from functools import reduce
import operator
import logging
def avg_int(lst):
    lst = list(lst)
    return sum(lst) // len(lst)