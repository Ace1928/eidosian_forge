import http.client
def tonative(n, encoding='ISO-8859-1'):
    """Return the given string as a native string in the given encoding."""
    if isinstance(n, bytes):
        return n.decode(encoding)
    return n