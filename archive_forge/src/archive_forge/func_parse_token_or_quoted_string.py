def parse_token_or_quoted_string(s, start=0, allow_quoted=True, allow_token=True):
    """Parses a token or a quoted-string.

    's' is the string to parse, while start is the position within the
    string where parsing should begin.  It will returns a tuple
    (token, chars_consumed), with all \\-escapes and quotation already
    processed.

    Syntax is according to BNF rules in RFC 2161 section 2.2,
    specifically the 'token' and 'quoted-string' declarations.
    Syntax errors in the input string will result in ParseError
    being raised.

    If allow_quoted is False, then only tokens will be parsed instead
    of either a token or quoted-string.

    If allow_token is False, then only quoted-strings will be parsed
    instead of either a token or quoted-string.
    """
    if not allow_quoted and (not allow_token):
        raise ValueError('Parsing can not continue with options provided')
    if start >= len(s):
        raise ParseError('Starting position is beyond the end of the string', s, start)
    has_quote = s[start] == '"'
    if has_quote and (not allow_quoted):
        raise ParseError('A quoted string was not expected', s, start)
    if not has_quote and (not allow_token):
        raise ParseError('Expected a quotation mark', s, start)
    s2 = ''
    pos = start
    if has_quote:
        pos += 1
    while pos < len(s):
        c = s[pos]
        if c == '\\' and has_quote:
            pos += 1
            if pos == len(s):
                raise ParseError("End of string while expecting a character after '\\'", s, pos)
            s2 += s[pos]
            pos += 1
        elif c == '"' and has_quote:
            break
        elif not has_quote and (c in SEPARATORS or ord(c) < 32 or ord(c) > 127):
            break
        else:
            s2 += c
            pos += 1
    if has_quote:
        if pos >= len(s) or s[pos] != '"':
            raise ParseError('Quoted string is missing closing quote mark', s, pos)
        else:
            pos += 1
    return (s2, pos - start)