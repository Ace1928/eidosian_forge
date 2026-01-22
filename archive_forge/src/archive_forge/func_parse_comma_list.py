def parse_comma_list(s, start=0, element_parser=None, min_count=0, max_count=0):
    """Parses a comma-separated list with optional whitespace.

    Takes an optional callback function `element_parser`, which
    is assumed to be able to parse an individual element.  It
    will be passed the string and a `start` argument, and
    is expected to return a tuple (parsed_result, chars_consumed).

    If no element_parser is given, then either single tokens or
    quoted strings will be parsed.

    If min_count > 0, then at least that many non-empty elements
    must be in the list, or an error is raised.

    If max_count > 0, then no more than that many non-empty elements
    may be in the list, or an error is raised.

    """
    if min_count > 0 and start == len(s):
        raise ParseError('Comma-separated list must contain some elements', s, start)
    elif start >= len(s):
        raise ParseError('Starting position is beyond the end of the string', s, start)
    if not element_parser:
        element_parser = parse_token_or_quoted_string
    results = []
    pos = start
    while pos < len(s):
        e = element_parser(s, pos)
        if not e or e[1] == 0:
            break
        else:
            results.append(e[0])
            pos += e[1]
        while pos < len(s) and s[pos] in LWS:
            pos += 1
        if pos < len(s) and s[pos] != ',':
            break
        while pos < len(s) and s[pos] == ',':
            pos += 1
            while pos < len(s) and s[pos] in LWS:
                pos += 1
    if len(results) < min_count:
        raise ParseError('Comma-separated list does not have enough elements', s, pos)
    elif max_count and len(results) > max_count:
        raise ParseError('Comma-separated list has too many elements', s, pos)
    return (results, pos - start)