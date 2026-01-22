import time, calendar
def mktime_tz(data):
    """Turn a 10-tuple as returned by parsedate_tz() into a POSIX timestamp."""
    if data[9] is None:
        return time.mktime(data[:8] + (-1,))
    else:
        t = calendar.timegm(data)
        return t - data[9]