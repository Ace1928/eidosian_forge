import collections
import pyrfc3339
from ._conditions import (
def time_before_caveat(t):
    """Return a caveat that specifies that the time that it is checked at
    should be before t.
    :param t is a a UTC date in - use datetime.utcnow, not datetime.now
    """
    return _first_party(COND_TIME_BEFORE, pyrfc3339.generate(t, accept_naive=True, microseconds=True))