from __future__ import unicode_literals
from datetime import datetime, time
def within_delta(dt1, dt2, delta):
    """
    Useful for comparing two datetimes that may have a negligible difference
    to be considered equal.
    """
    delta = abs(delta)
    difference = dt1 - dt2
    return -delta <= difference <= delta