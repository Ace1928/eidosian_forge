import re
from isodate.isoerror import ISO8601Error
from isodate.tzinfo import UTC, FixedOffset, ZERO
def parse_tzinfo(tzstring):
    """
    Parses ISO 8601 time zone designators to tzinfo objecs.

    A time zone designator can be in the following format:
              no designator indicates local time zone
      Z       UTC
      +-hhmm  basic hours and minutes
      +-hh:mm extended hours and minutes
      +-hh    hours
    """
    match = TZ_RE.match(tzstring)
    if match:
        groups = match.groupdict()
        return build_tzinfo(groups['tzname'], groups['tzsign'], int(groups['tzhour'] or 0), int(groups['tzmin'] or 0))
    raise ISO8601Error('%s not a valid time zone info' % tzstring)