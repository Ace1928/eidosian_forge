import sys
from functools import partial
from pydev_ipython.version import check_version
def qtapi_version():
    """Return which QString API has been set, if any

    Returns
    -------
    The QString API version (1 or 2), or None if not set
    """
    try:
        import sip
    except ImportError:
        return
    try:
        return sip.getapi('QString')
    except ValueError:
        return