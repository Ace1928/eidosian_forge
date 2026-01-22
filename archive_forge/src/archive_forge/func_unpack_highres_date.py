import calendar
import re
import time
from . import osutils
def unpack_highres_date(date):
    """This takes the high-resolution date stamp, and
    converts it back into the tuple (timestamp, timezone)
    Where timestamp is in real UTC since epoch seconds, and timezone is an
    integer number of seconds offset.

    :param date: A date formated by format_highres_date
    :type date: string

    """
    space_loc = date.find(' ')
    if space_loc == -1 or date[:space_loc] not in osutils.weekdays:
        raise ValueError('Date string does not contain a day of week: %r' % date)
    dot_loc = date.find('.')
    if dot_loc == -1:
        raise ValueError('Date string does not contain high-precision seconds: %r' % date)
    base_time = time.strptime(date[space_loc:dot_loc], ' %Y-%m-%d %H:%M:%S')
    fract_seconds, offset = date[dot_loc:].split()
    fract_seconds = float(fract_seconds)
    offset = int(offset)
    hours = int(offset / 100)
    minutes = offset % 100
    seconds_offset = hours * 3600 + minutes * 60
    timestamp = calendar.timegm(base_time)
    timestamp -= seconds_offset
    timestamp += fract_seconds
    return (timestamp, seconds_offset)