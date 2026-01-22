import calendar
import re
import time
from . import osutils
def parse_patch_date(date_str):
    """Parse a patch-style date into a POSIX timestamp and offset.

    Inverse of format_patch_date.
    """
    match = RE_PATCHDATE.match(date_str)
    if match is None:
        if RE_PATCHDATE_NOOFFSET.match(date_str) is not None:
            raise ValueError('time data %r is missing a timezone offset' % date_str)
        else:
            raise ValueError('time data %r does not match format ' % date_str + "'%Y-%m-%d %H:%M:%S %z'")
    secs_str = match.group(1)
    offset_hours, offset_mins = (int(match.group(2)), int(match.group(3)))
    if abs(offset_hours) >= 24 or offset_mins >= 60:
        raise ValueError('invalid timezone %r' % (match.group(2) + match.group(3)))
    offset = offset_hours * 3600 + offset_mins * 60
    tm_time = time.strptime(secs_str, '%Y-%m-%d %H:%M:%S')
    tm_time = tm_time[:5] + (tm_time[5] - offset,) + tm_time[6:]
    secs = calendar.timegm(tm_time)
    return (secs, offset)