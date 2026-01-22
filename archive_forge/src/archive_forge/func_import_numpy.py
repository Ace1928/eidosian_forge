import sys
def import_numpy():
    """
    Returns the numpy module if it exists on the system,
    otherwise returns None.
    """
    if PY3:
        numpy_exists = importlib.machinery.PathFinder.find_spec('numpy') is not None
    else:
        try:
            imp.find_module('numpy')
            numpy_exists = True
        except ImportError:
            numpy_exists = False
    if numpy_exists:
        import numpy as np
    else:
        np = None
    return np