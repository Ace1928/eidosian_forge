from datetime import datetime
from isodate.isostrf import strftime
from isodate.isostrf import DATE_EXT_COMPLETE, TIME_EXT_COMPLETE, TZ_EXT
from isodate.isodates import parse_date
from isodate.isoerror import ISO8601Error
from isodate.isotime import parse_time

    Format datetime strings.

    This method is just a wrapper around isodate.isostrf.strftime and uses
    Extended-Complete as default format.
    