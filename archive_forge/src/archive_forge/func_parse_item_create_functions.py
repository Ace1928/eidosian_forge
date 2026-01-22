def parse_item_create_functions(f):
    """Parse MySQL function names from item_create.cc."""
    results = set()
    for m in re.finditer('{"(?P<function>[^"]+?)",\\s*SQL_F[^(]+?\\(', f, flags=re.I):
        results.add(m.group('function').lower())
    if not results:
        raise ValueError('No item_create functions found')
    return results