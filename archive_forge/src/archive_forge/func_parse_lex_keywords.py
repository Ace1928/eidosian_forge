def parse_lex_keywords(f):
    """Parse keywords in lex.h."""
    results = set()
    for m in re.finditer('{SYM(?:_HK)?\\("(?P<keyword>[a-z0-9_]+)",', f, flags=re.I):
        results.add(m.group('keyword').lower())
    if not results:
        raise ValueError('No keywords found')
    return results