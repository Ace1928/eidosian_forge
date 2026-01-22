from __future__ import print_function, unicode_literals
import typing
from calendar import timegm
from datetime import datetime
def datetime_to_epoch(d):
    """Convert datetime to epoch."""
    return timegm(d.utctimetuple())