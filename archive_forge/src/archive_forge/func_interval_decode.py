import re
def interval_decode(s):
    """Decodes a number in the format 1h4d3m (1 hour, 3 days, 3 minutes)
    into a number of seconds

    >>> interval_decode('40s')
    40
    >>> interval_decode('10000s')
    10000
    >>> interval_decode('3d1w45s')
    864045
    """
    time = 0
    sign = 1
    s = s.strip()
    if s.startswith('-'):
        s = s[1:]
        sign = -1
    elif s.startswith('+'):
        s = s[1:]
    for match in allMatches(s, _timeRE):
        char = match.group(0)[-1].lower()
        if char not in timeValues:
            continue
        time += int(match.group(0)[:-1]) * timeValues[char]
    return time