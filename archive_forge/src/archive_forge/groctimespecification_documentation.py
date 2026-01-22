from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import calendar
import datetime
from . import groc
Returns the next match after time start.

    Must be implemented in subclasses.

    Arguments:
      start: a datetime to start from. Matches will start from after this time.
        This may be in any pytz time zone, or it may be timezone-naive
        (interpreted as UTC).

    Returns:
      a datetime object in the timezone of the input 'start'
    