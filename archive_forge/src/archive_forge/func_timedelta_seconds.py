from __future__ import division
from datetime import timedelta, tzinfo
from copy import deepcopy
def timedelta_seconds(td):
    """
    Return the offset stored by a :class:`datetime.timedelta` object as an
    integer number of seconds.  Microseconds, if present, are rounded to
    the nearest second.

    Delegates to
    :meth:`timedelta.total_seconds() <datetime.timedelta.total_seconds()>`
    if available.

    >>> timedelta_seconds(timedelta(hours=1))
    3600
    >>> timedelta_seconds(timedelta(hours=-1))
    -3600
    >>> timedelta_seconds(timedelta(hours=1, minutes=30))
    5400
    >>> timedelta_seconds(timedelta(hours=1, minutes=30,
    ... microseconds=300000))
    5400
    >>> timedelta_seconds(timedelta(hours=1, minutes=30,
    ...	microseconds=900000))
    5401

    """
    try:
        return int(round(td.total_seconds()))
    except AttributeError:
        days = td.days
        seconds = td.seconds
        microseconds = td.microseconds
        return int(round(days * 86400 + seconds + microseconds / 1000000))