import datetime
import logging
import iso8601
def from_utcnow(**timedelta_kwargs):
    """Calculate the time in the future from utcnow.

    :param \\*\\*timedelta_kwargs:
        Passed directly to :class:`datetime.timedelta` to add to the current
        time in UTC.
    :returns:
        The time in the future based on ``timedelta_kwargs``.
    :rtype:
        datetime.datetime
    """
    now = datetime.datetime.utcnow()
    delta = datetime.timedelta(**timedelta_kwargs)
    return now + delta