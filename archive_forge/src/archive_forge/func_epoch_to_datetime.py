from __future__ import print_function, unicode_literals
import typing
from calendar import timegm
from datetime import datetime
def epoch_to_datetime(t):
    """Convert epoch time to a UTC datetime."""
    if t is None:
        return None
    return datetime.fromtimestamp(t, tz=timezone.utc)