from __future__ import absolute_import, print_function, division
import datetime
from petl.compat import long
def datetimeparser(fmt, strict=True):
    """Return a function to parse strings as :class:`datetime.datetime` objects
    using a given format. E.g.::

        >>> from petl import datetimeparser
        >>> isodatetime = datetimeparser('%Y-%m-%dT%H:%M:%S')
        >>> isodatetime('2002-12-25T00:00:00')
        datetime.datetime(2002, 12, 25, 0, 0)
        >>> try:
        ...     isodatetime('2002-12-25T00:00:99')
        ... except ValueError as e:
        ...     print(e)
        ...
        unconverted data remains: 9

    If ``strict=False`` then if an error occurs when parsing, the original
    value will be returned as-is, and no error will be raised.

    """

    def parser(value):
        try:
            return datetime.datetime.strptime(value.strip(), fmt)
        except Exception as e:
            if strict:
                raise e
            else:
                return value
    return parser