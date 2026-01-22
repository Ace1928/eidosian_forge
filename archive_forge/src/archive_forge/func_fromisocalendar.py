import time as _time
import math as _math
import sys
from operator import index as _index
@classmethod
def fromisocalendar(cls, year, week, day):
    """Construct a date from the ISO year, week number and weekday.

        This is the inverse of the date.isocalendar() function"""
    return cls(*_isoweek_to_gregorian(year, week, day))