import calendar
import copy
import datetime
import re
import sys
import time
import types
from dateutil import parser
import pytz
@classmethod
def fromtimestamp(cls, *args, **kwargs):
    """Get a new localized timestamp from a POSIX timestamp.

    Args:
      args: Positional arguments to pass to datetime.datetime.fromtimestamp().
      kwargs: Keyword arguments to pass to datetime.datetime.fromtimestamp().
              If tz is not specified, local timezone is assumed.

    Returns:
      A new BaseTimestamp with tz's local day and time.
    """
    return cls.Localize(super(BaseTimestamp, cls).fromtimestamp(*args, **kwargs))