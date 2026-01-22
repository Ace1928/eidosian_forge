import functools
@property
def compressed(self):
    """Return the shorthand version of the IP address as a string."""
    return str(self)