import collections
import datetime
import decimal
import numbers
import os
import os.path
import re
import time
from humanfriendly.compat import is_string, monotonic
from humanfriendly.deprecation import define_aliases
from humanfriendly.text import concatenate, format, pluralize, tokenize
def coerce_seconds(value):
    """
    Coerce a value to the number of seconds.

    :param value: An :class:`int`, :class:`float` or
                  :class:`datetime.timedelta` object.
    :returns: An :class:`int` or :class:`float` value.

    When `value` is a :class:`datetime.timedelta` object the
    :meth:`~datetime.timedelta.total_seconds()` method is called.
    """
    if isinstance(value, datetime.timedelta):
        return value.total_seconds()
    if not isinstance(value, numbers.Number):
        msg = 'Failed to coerce value to number of seconds! (%r)'
        raise ValueError(format(msg, value))
    return value