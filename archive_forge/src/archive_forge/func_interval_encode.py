import re
def interval_encode(seconds, include_sign=False):
    """Encodes a number of seconds (representing a time interval)
    into a form like 1h2d3s.

    >>> interval_encode(10)
    '10s'
    >>> interval_encode(493939)
    '5d17h12m19s'
    """
    s = ''
    orig = seconds
    seconds = abs(seconds)
    for char, amount in timeOrdered:
        if seconds >= amount:
            i, seconds = divmod(seconds, amount)
            s += '%i%s' % (i, char)
    if orig < 0:
        s = '-' + s
    elif not orig:
        return '0'
    elif include_sign:
        s = '+' + s
    return s