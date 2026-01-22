def parse_token(s, start=0):
    """Parses a token.

    A token is a string defined by RFC 2616 section 2.2 as:
       token = 1*<any CHAR except CTLs or separators>

    Returns a tuple (token, chars_consumed), or ('',0) if no token
    starts at the given string position.  On a syntax error, a
    ParseError exception will be raised.

    """
    return parse_token_or_quoted_string(s, start, allow_quoted=False, allow_token=True)