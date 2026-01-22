import datetime
import iso8601
from oslo_utils import encodeutils
def delta_seconds(before, after):
    """Return the difference between two timing objects.

    Compute the difference in seconds between two date, time, or
    datetime objects (as a float, to microsecond resolution).
    """
    delta = after - before
    return datetime.timedelta.total_seconds(delta)