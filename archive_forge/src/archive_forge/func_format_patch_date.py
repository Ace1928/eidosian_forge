import calendar
import re
import time
from . import osutils
def format_patch_date(secs, offset=0):
    """Format a POSIX timestamp and optional offset as a patch-style date.

    Inverse of parse_patch_date.
    """
    if offset % 60 != 0:
        raise ValueError("can't represent timezone %s offset by fractional minutes" % offset)
    if secs == 0:
        offset = 0
    if secs + offset < 0:
        from warnings import warn
        warn('gmtime of negative time (%s, %s) may not work on Windows' % (secs, offset))
    return osutils.format_date(secs, offset=offset, date_fmt='%Y-%m-%d %H:%M:%S')