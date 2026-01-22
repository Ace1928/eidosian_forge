def pylong_join(count, digits_ptr='digits', join_type='unsigned long'):
    """
    Generate an unrolled shift-then-or loop over the first 'count' digits.
    Assumes that they fit into 'join_type'.

    (((d[2] << n) | d[1]) << n) | d[0]
    """
    return '(' * (count * 2) + ' | '.join(('(%s)%s[%d])%s)' % (join_type, digits_ptr, _i, ' << PyLong_SHIFT' if _i else '') for _i in range(count - 1, -1, -1)))