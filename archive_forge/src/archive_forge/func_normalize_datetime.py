import re
from datetime import datetime
def normalize_datetime(dtstr, match=None):
    """Try to normalize a datetime string.
    1. Convert 12-hour time to 24-hour time

    pass match in if we have already calculated it to avoid rework
    """
    match = match or (dtstr and re.match(DATETIME_RE + '$', dtstr))
    if match:
        datestr = match.group('date')
        hourstr = match.group('hour')
        minutestr = match.group('minute') or '00'
        secondstr = match.group('second')
        ampmstr = match.group('ampm')
        separator = match.group('separator')
        try:
            datestr = datetime.strptime(datestr, '%Y-%j').strftime('%Y-%m-%d')
        except ValueError:
            pass
        if ampmstr:
            hourstr = match.group('hour')
            hourint = int(hourstr)
            if (ampmstr.startswith('a') or ampmstr.startswith('A')) and hourint == 12:
                hourstr = '00'
            if (ampmstr.startswith('p') or ampmstr.startswith('P')) and hourint < 12:
                hourstr = hourint + 12
        dtstr = '%s%s%s:%s' % (datestr, separator, hourstr, minutestr)
        if secondstr:
            dtstr += ':' + secondstr
        tzstr = match.group('tz')
        if tzstr:
            dtstr += tzstr.replace(':', '')
    return dtstr