from six import PY2
from functools import wraps
from datetime import datetime, timedelta, tzinfo
def enfold(dt, fold=1):
    """
        Provides a unified interface for assigning the ``fold`` attribute to
        datetimes both before and after the implementation of PEP-495.

        :param fold:
            The value for the ``fold`` attribute in the returned datetime. This
            should be either 0 or 1.

        :return:
            Returns an object for which ``getattr(dt, 'fold', 0)`` returns
            ``fold`` for all versions of Python. In versions prior to
            Python 3.6, this is a ``_DatetimeWithFold`` object, which is a
            subclass of :py:class:`datetime.datetime` with the ``fold``
            attribute added, if ``fold`` is 1.

        .. versionadded:: 2.6.0
        """
    if getattr(dt, 'fold', 0) == fold:
        return dt
    args = dt.timetuple()[:6]
    args += (dt.microsecond, dt.tzinfo)
    if fold:
        return _DatetimeWithFold(*args)
    else:
        return datetime(*args)