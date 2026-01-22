def parse_http_datetime(datestring, utc_tzinfo=None, strict=False):
    """Returns a datetime object from an HTTP 1.1 Date/Time string.

    Note that HTTP dates are always in UTC, so the returned datetime
    object will also be in UTC.

    You can optionally pass in a tzinfo object which should represent
    the UTC timezone, and the returned datetime will then be
    timezone-aware (allowing you to more easly translate it into
    different timzeones later).

    If you set 'strict' to True, then only the RFC 1123 format
    is recognized.  Otherwise the backwards-compatible RFC 1036
    and Unix asctime(3) formats are also recognized.
    
    Please note that the day-of-the-week is not validated.
    Also two-digit years, although not HTTP 1.1 compliant, are
    treated according to recommended Y2K rules.

    """
    import re, datetime
    m = re.match('(?P<DOW>[a-z]+), (?P<D>\\d+) (?P<MON>[a-z]+) (?P<Y>\\d+) (?P<H>\\d+):(?P<M>\\d+):(?P<S>\\d+(\\.\\d+)?) (?P<TZ>[a-zA-Z0-9_+]+)$', datestring, re.IGNORECASE)
    if not m and (not strict):
        m = re.match('(?P<DOW>[a-z]+) (?P<MON>[a-z]+) (?P<D>\\d+) (?P<H>\\d+):(?P<M>\\d+):(?P<S>\\d+) (?P<Y>\\d+)$', datestring, re.IGNORECASE)
        if not m:
            m = re.match('(?P<DOW>[a-z]+), (?P<D>\\d+)-(?P<MON>[a-z]+)-(?P<Y>\\d+) (?P<H>\\d+):(?P<M>\\d+):(?P<S>\\d+(\\.\\d+)?) (?P<TZ>\\w+)$', datestring, re.IGNORECASE)
    if not m:
        raise ValueError('HTTP date is not correctly formatted')
    try:
        tz = m.group('TZ').upper()
    except:
        tz = 'GMT'
    if tz not in ('GMT', 'UTC', '0000', '00:00'):
        raise ValueError('HTTP date is not in GMT timezone')
    monname = m.group('MON').upper()
    mdict = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
    month = mdict.get(monname)
    if not month:
        raise ValueError('HTTP date has an unrecognizable month')
    y = int(m.group('Y'))
    if y < 100:
        century = datetime.datetime.utcnow().year / 100
        if y < 50:
            y = century * 100 + y
        else:
            y = (century - 1) * 100 + y
    d = int(m.group('D'))
    hour = int(m.group('H'))
    minute = int(m.group('M'))
    try:
        second = int(m.group('S'))
    except:
        second = float(m.group('S'))
    dt = datetime.datetime(y, month, d, hour, minute, second, tzinfo=utc_tzinfo)
    return dt