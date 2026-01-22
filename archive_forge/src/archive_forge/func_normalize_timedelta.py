from datetime import timedelta, time, date
from time import localtime
def normalize_timedelta(val):
    """
    produces a normalized string value of the timedelta

    This module returns a normalized time span value consisting of the
    number of hours in fractional form. For example '1h 15min' is
    formatted as 01.25.
    """
    if type(val) == str:
        val = parse_timedelta(val)
    if not val:
        return ''
    hr = val.seconds / 3600
    mn = val.seconds % 3600 / 60
    return '%d.%02d' % (hr, mn * 100 / 60)