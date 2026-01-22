import calendar
import re
import time
from . import osutils
def format_highres_date(t, offset=0):
    """Format a date, such that it includes higher precision in the
    seconds field.

    :param t:   The local time in fractional seconds since the epoch
    :type t: float
    :param offset:  The timezone offset in integer seconds
    :type offset: int

    Example: format_highres_date(time.time(), -time.timezone)
    this will return a date stamp for right now,
    formatted for the local timezone.

    >>> from breezy.osutils import format_date
    >>> format_date(1120153132.350850105, 0)
    'Thu 2005-06-30 17:38:52 +0000'
    >>> format_highres_date(1120153132.350850105, 0)
    'Thu 2005-06-30 17:38:52.350850105 +0000'
    >>> format_date(1120153132.350850105, -5*3600)
    'Thu 2005-06-30 12:38:52 -0500'
    >>> format_highres_date(1120153132.350850105, -5*3600)
    'Thu 2005-06-30 12:38:52.350850105 -0500'
    >>> format_highres_date(1120153132.350850105, 7200)
    'Thu 2005-06-30 19:38:52.350850105 +0200'
    >>> format_highres_date(1152428738.867522, 19800)
    'Sun 2006-07-09 12:35:38.867522001 +0530'
    """
    if not isinstance(t, float):
        raise ValueError(t)
    if offset is None:
        offset = 0
    tt = time.gmtime(t + offset)
    return osutils.weekdays[tt[6]] + time.strftime(' %Y-%m-%d %H:%M:%S', tt) + ('%.9f' % (t - int(t)))[1:] + ' %+03d%02d' % (offset / 3600, offset / 60 % 60)