from datetime import datetime, timedelta, tzinfo
from bisect import bisect_right
import pytz
from pytz.exceptions import AmbiguousTimeError, NonExistentTimeError
def memorized_ttinfo(*args):
    """Create only one instance of each distinct tuple"""
    try:
        return _ttinfo_cache[args]
    except KeyError:
        ttinfo = (memorized_timedelta(args[0]), memorized_timedelta(args[1]), args[2])
        _ttinfo_cache[args] = ttinfo
        return ttinfo