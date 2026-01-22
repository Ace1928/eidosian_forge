import re
from calendar import day_abbr, day_name, month_abbr, month_name
from datetime import datetime as datetime_
from datetime import timedelta, timezone
from time import localtime, strftime
def aware_now():
    now = datetime_.now()
    timestamp = now.timestamp()
    local = localtime(timestamp)
    try:
        seconds = local.tm_gmtoff
        zone = local.tm_zone
    except AttributeError:
        utc_naive = datetime_.fromtimestamp(timestamp, tz=timezone.utc).replace(tzinfo=None)
        offset = datetime_.fromtimestamp(timestamp) - utc_naive
        seconds = offset.total_seconds()
        zone = strftime('%Z')
    tzinfo = timezone(timedelta(seconds=seconds), zone)
    return datetime.combine(now.date(), now.time().replace(tzinfo=tzinfo))