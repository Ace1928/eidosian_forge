def http_datetime(dt=None):
    """Formats a datetime as an HTTP 1.1 Date/Time string.

    Takes a standard Python datetime object and returns a string
    formatted according to the HTTP 1.1 date/time format.

    If no datetime is provided (or None) then the current
    time is used.
    
    ABOUT TIMEZONES: If the passed in datetime object is naive it is
    assumed to be in UTC already.  But if it has a tzinfo component,
    the returned timestamp string will have been converted to UTC
    automatically.  So if you use timezone-aware datetimes, you need
    not worry about conversion to UTC.

    """
    if not dt:
        import datetime
        dt = datetime.datetime.utcnow()
    else:
        try:
            dt = dt - dt.utcoffset()
        except:
            pass
    s = dt.strftime('%a, %d %b %Y %H:%M:%S GMT')
    return s