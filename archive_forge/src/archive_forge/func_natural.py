from calendar import timegm
from datetime import datetime, time, timedelta
from email.utils import parsedate_tz, mktime_tz
import re
import aniso8601
import pytz
def natural(value, argument='argument'):
    """ Restrict input type to the natural numbers (0, 1, 2, 3...) """
    value = _get_integer(value)
    if value < 0:
        error = 'Invalid {arg}: {value}. {arg} must be a non-negative integer'.format(arg=argument, value=value)
        raise ValueError(error)
    return value