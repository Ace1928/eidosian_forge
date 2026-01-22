def invert_dict(d):
    """Invert a dictionary with keys matching each value turned into a list."""
    result = {}
    for k, v in d.items():
        keys = result.setdefault(v, [])
        keys.append(k)
    return result