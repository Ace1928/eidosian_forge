import datetime
import json
from functools import wraps
from io import StringIO
from nltk.twitter import (
def yesterday():
    """
    Get yesterday's datetime as a 5-tuple.
    """
    date = datetime.datetime.now()
    date -= datetime.timedelta(days=1)
    date_tuple = date.timetuple()[:6]
    return date_tuple