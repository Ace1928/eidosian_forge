def parse_range_set(s, start=0, valid_units=('bytes', 'none')):
    """Parses a (byte) range set specifier.

    Returns a tuple (range_set, chars_consumed).
    """
    if start >= len(s):
        raise ParseError('Starting position is beyond the end of the string', s, start)
    pos = start
    units, k = parse_token(s, pos)
    pos += k
    if valid_units and units not in valid_units:
        raise ParseError('Unsupported units type in range specifier', s, start)
    while pos < len(s) and s[pos] in LWS:
        pos += 1
    if pos < len(s) and s[pos] == '=':
        pos += 1
    else:
        raise ParseError("Invalid range specifier, expected '='", s, pos)
    while pos < len(s) and s[pos] in LWS:
        pos += 1
    range_specs, k = parse_comma_list(s, pos, parse_range_spec, min_count=1)
    pos += k
    while pos < len(s) and s[pos] in LWS:
        pos += 1
    if pos < len(s):
        raise ParseError('Unparsable characters in range set specifier', s, pos)
    ranges = range_set()
    ranges.units = units
    ranges.range_specs = range_specs
    return (ranges, pos - start)