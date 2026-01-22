import sys
def dict_subtract(a, b):
    """Return the part of ``a`` that's not in ``b``."""
    return {k: a[k] for k in set(a) - set(b)}