def parse_quoted_string(s, start=0):
    """Parses a quoted string.

    Returns a tuple (string, chars_consumed).  The quote marks will
    have been removed and all \\-escapes will have been replaced with
    the characters they represent.

    """
    return parse_token_or_quoted_string(s, start, allow_quoted=True, allow_token=False)