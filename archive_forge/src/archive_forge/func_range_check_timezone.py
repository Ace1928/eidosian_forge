import calendar
from collections import namedtuple
from aniso8601.exceptions import (
@classmethod
def range_check_timezone(cls, negative=None, Z=None, hh=None, mm=None, name='', rangedict=None):
    if rangedict is None:
        rangedict = cls.TIMEZONE_RANGE_DICT
    if 'hh' in rangedict:
        hh = rangedict['hh'].rangefunc(hh, rangedict['hh'])
    if 'mm' in rangedict:
        mm = rangedict['mm'].rangefunc(mm, rangedict['mm'])
    return (negative, Z, hh, mm, name)