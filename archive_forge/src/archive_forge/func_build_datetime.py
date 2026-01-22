import calendar
from collections import namedtuple
from aniso8601.exceptions import (
@classmethod
def build_datetime(cls, date, time):
    return DatetimeTuple(date, time)