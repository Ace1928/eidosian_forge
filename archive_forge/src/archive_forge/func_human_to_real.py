from __future__ import absolute_import, division, print_function
def human_to_real(iops):
    """Given a human-readable string (e.g. 2K, 30M IOPs),
    return the real number.  Will return 0 if the argument has
    unexpected form.
    """
    digit = iops[:-1]
    unit = iops[-1].upper()
    if unit.isdigit():
        digit = iops
    elif digit.isdigit():
        digit = int(digit)
        if unit == 'M':
            digit *= 1000000
        elif unit == 'K':
            digit *= 1000
        else:
            digit = 0
    else:
        digit = 0
    return digit