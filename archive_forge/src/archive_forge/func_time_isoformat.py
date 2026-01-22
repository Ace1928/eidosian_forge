import re
from decimal import Decimal
from datetime import time
from isodate.isostrf import strftime, TIME_EXT_COMPLETE, TZ_EXT
from isodate.isoerror import ISO8601Error
from isodate.isotzinfo import TZ_REGEX, build_tzinfo
def time_isoformat(ttime, format=TIME_EXT_COMPLETE + TZ_EXT):
    """
    Format time strings.

    This method is just a wrapper around isodate.isostrf.strftime and uses
    Time-Extended-Complete with extended time zone as default format.
    """
    return strftime(ttime, format)