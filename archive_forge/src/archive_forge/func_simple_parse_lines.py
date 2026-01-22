def simple_parse_lines(lines):
    """Same as simple_parse, but takes an iterable of strs rather than a single
    str.
    """
    return simple_parse(''.join(lines))