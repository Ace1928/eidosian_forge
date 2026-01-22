import calendar
from collections import namedtuple
from aniso8601.exceptions import (
@classmethod
def range_check_repeating_interval(cls, R=None, Rnn=None, interval=None, rangedict=None):
    if rangedict is None:
        rangedict = cls.REPEATING_INTERVAL_RANGE_DICT
    if 'Rnn' in rangedict:
        Rnn = rangedict['Rnn'].rangefunc(Rnn, rangedict['Rnn'])
    return (R, Rnn, interval)