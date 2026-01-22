def methodkey(self, *args, **kwargs):
    """Return a cache key for use with cached methods."""
    return hashkey(*args, **kwargs)