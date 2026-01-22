import re
from decimal import Decimal
from datetime import time
from isodate.isostrf import strftime, TIME_EXT_COMPLETE, TZ_EXT
from isodate.isoerror import ISO8601Error
from isodate.isotzinfo import TZ_REGEX, build_tzinfo
def build_time_regexps():
    """
    Build regular expressions to parse ISO time string.

    The regular expressions are compiled and stored in TIME_REGEX_CACHE
    for later reuse.
    """
    if not TIME_REGEX_CACHE:
        TIME_REGEX_CACHE.append(re.compile('T?(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2}([,.][0-9]+)?)' + TZ_REGEX))
        TIME_REGEX_CACHE.append(re.compile('T?(?P<hour>[0-9]{2})(?P<minute>[0-9]{2})(?P<second>[0-9]{2}([,.][0-9]+)?)' + TZ_REGEX))
        TIME_REGEX_CACHE.append(re.compile('T?(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}([,.][0-9]+)?)' + TZ_REGEX))
        TIME_REGEX_CACHE.append(re.compile('T?(?P<hour>[0-9]{2})(?P<minute>[0-9]{2}([,.][0-9]+)?)' + TZ_REGEX))
        TIME_REGEX_CACHE.append(re.compile('T?(?P<hour>[0-9]{2}([,.][0-9]+)?)' + TZ_REGEX))
    return TIME_REGEX_CACHE