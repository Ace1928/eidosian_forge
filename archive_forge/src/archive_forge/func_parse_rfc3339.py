import datetime
import math
import re
def parse_rfc3339(s):
    if isinstance(s, datetime.datetime):
        if not s.tzinfo:
            return s.replace(tzinfo=UTC)
        return s
    groups = _re_rfc3339.search(s).groups()
    dt = [0] * 7
    for x in range(6):
        dt[x] = int(groups[x])
    if groups[6] is not None:
        dt[6] = int(groups[6])
    tz = UTC
    if groups[7] is not None and groups[7] != 'Z' and (groups[7] != 'z'):
        tz_groups = _re_timezone.search(groups[7]).groups()
        hour = int(tz_groups[1])
        minute = 0
        if tz_groups[0] == '-':
            hour *= -1
        if tz_groups[2]:
            minute = int(tz_groups[2])
        tz = TimezoneInfo(hour, minute)
    return datetime.datetime(year=dt[0], month=dt[1], day=dt[2], hour=dt[3], minute=dt[4], second=dt[5], microsecond=dt[6], tzinfo=tz)