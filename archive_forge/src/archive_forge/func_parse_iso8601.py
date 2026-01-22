import re
from datetime import datetime, timedelta, timezone
from celery.utils.deprecated import warn
def parse_iso8601(datestring: str) -> datetime:
    """Parse and convert ISO-8601 string to datetime."""
    warn('parse_iso8601', 'v5.3', 'v6', 'datetime.datetime.fromisoformat or dateutil.parser.isoparse')
    m = ISO8601_REGEX.match(datestring)
    if not m:
        raise ValueError('unable to parse date string %r' % datestring)
    groups = m.groupdict()
    tz = groups['timezone']
    if tz == 'Z':
        tz = timezone(timedelta(0))
    elif tz:
        m = TIMEZONE_REGEX.match(tz)
        prefix, hours, minutes = m.groups()
        hours, minutes = (int(hours), int(minutes))
        if prefix == '-':
            hours = -hours
            minutes = -minutes
        tz = timezone(timedelta(minutes=minutes, hours=hours))
    return datetime(int(groups['year']), int(groups['month']), int(groups['day']), int(groups['hour'] or 0), int(groups['minute'] or 0), int(groups['second'] or 0), int(groups['fraction'] or 0), tz)