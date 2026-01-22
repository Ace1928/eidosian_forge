from __future__ import unicode_literals
import functools
import re
from datetime import timedelta
import logging
import io
def srt_timestamp_to_timedelta(timestamp):
    """
    Convert an SRT timestamp to a :py:class:`~datetime.timedelta`.

    .. doctest::

        >>> srt_timestamp_to_timedelta('01:23:04,000')
        datetime.timedelta(seconds=4984)

    :param str timestamp: A timestamp in SRT format
    :returns: The timestamp as a :py:class:`~datetime.timedelta`
    :rtype: datetime.timedelta
    :raises TimestampParseError: If the timestamp is not parseable
    """
    match = TS_REGEX.match(timestamp)
    if match is None:
        raise TimestampParseError('Unparseable timestamp: {}'.format(timestamp))
    hrs, mins, secs, msecs = [int(m) if m else 0 for m in match.groups()]
    return timedelta(hours=hrs, minutes=mins, seconds=secs, milliseconds=msecs)