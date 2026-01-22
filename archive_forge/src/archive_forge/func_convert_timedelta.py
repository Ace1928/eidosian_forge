import datetime
from decimal import Decimal
import re
import time
from .err import ProgrammingError
from .constants import FIELD_TYPE
def convert_timedelta(obj):
    """Returns a TIME column as a timedelta object:

      >>> convert_timedelta('25:06:17')
      datetime.timedelta(days=1, seconds=3977)
      >>> convert_timedelta('-25:06:17')
      datetime.timedelta(days=-2, seconds=82423)

    Illegal values are returned as string:

      >>> convert_timedelta('random crap')
      'random crap'

    Note that MySQL always returns TIME columns as (+|-)HH:MM:SS, but
    can accept values as (+|-)DD HH:MM:SS. The latter format will not
    be parsed correctly by this function.
    """
    if isinstance(obj, (bytes, bytearray)):
        obj = obj.decode('ascii')
    m = TIMEDELTA_RE.match(obj)
    if not m:
        return obj
    try:
        groups = list(m.groups())
        groups[-1] = _convert_second_fraction(groups[-1])
        negate = -1 if groups[0] else 1
        hours, minutes, seconds, microseconds = groups[1:]
        tdelta = datetime.timedelta(hours=int(hours), minutes=int(minutes), seconds=int(seconds), microseconds=int(microseconds)) * negate
        return tdelta
    except ValueError:
        return obj