import time
from . import errors
def parse_raw(s, lineno=0):
    """Parse a date from a raw string.

    The format must be exactly "seconds-since-epoch offset-utc".
    See the spec for details.
    """
    timestamp_str, timezone_str = s.split(b' ', 1)
    timestamp = float(timestamp_str)
    try:
        timezone = parse_tz(timezone_str)
    except ValueError:
        raise errors.InvalidTimezone(lineno, timezone_str)
    return (timestamp, timezone)