def parse_range_spec(s, start=0):
    """Parses a (byte) range_spec.

    Returns a tuple (range_spec, chars_consumed).
    """
    if start >= len(s):
        raise ParseError('Starting position is beyond the end of the string', s, start)
    if s[start] not in DIGIT and s[start] != '-':
        raise ParseError("Invalid range, expected a digit or '-'", s, start)
    _first, last = (None, None)
    pos = start
    first, k = parse_number(s, pos)
    pos += k
    if s[pos] == '-':
        pos += 1
        if pos < len(s):
            last, k = parse_number(s, pos)
            pos += k
    else:
        raise ParseError("Byte range must include a '-'", s, pos)
    if first is None and last is None:
        raise ParseError('Byte range can not omit both first and last indices.', s, start)
    R = range_spec(first, last)
    return (R, pos - start)