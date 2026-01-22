from __future__ import absolute_import, print_function, division
import datetime
from petl.compat import long
def timeparser(fmt, strict=True):
    """Return a function to parse strings as :class:`datetime.time` objects
    using a given format. E.g.::

        >>> from petl import timeparser
        >>> isotime = timeparser('%H:%M:%S')
        >>> isotime('00:00:00')
        datetime.time(0, 0)
        >>> isotime('13:00:00')
        datetime.time(13, 0)
        >>> try:
        ...     isotime('12:00:99')
        ... except ValueError as e:
        ...     print(e)
        ...
        unconverted data remains: 9
        >>> try:
        ...     isotime('25:00:00')
        ... except ValueError as e:
        ...     print(e)
        ...
        time data '25:00:00' does not match format '%H:%M:%S'

    If ``strict=False`` then if an error occurs when parsing, the original
    value will be returned as-is, and no error will be raised.

    """

    def parser(value):
        try:
            return datetime.datetime.strptime(value.strip(), fmt).time()
        except Exception as e:
            if strict:
                raise e
            else:
                return value
    return parser