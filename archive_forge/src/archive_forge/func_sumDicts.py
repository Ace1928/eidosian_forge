from fontTools.misc.timeTools import timestampNow
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from functools import reduce
import operator
import logging
def sumDicts(lst):
    d = {}
    for item in lst:
        d.update(item)
    return d