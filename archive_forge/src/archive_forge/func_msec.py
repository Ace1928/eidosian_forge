import datetime
import logging
from urllib import parse as urlparse
from osprofiler import _utils
def msec(dt):
    microsec = dt.microseconds + (dt.seconds + dt.days * 24 * 3600) * 1000000.0
    return int(microsec / 1000.0)