import sys
import os
from warnings import warn
from .compat import filename_decode, filename_encode
from .base import phil, with_phil
from .group import Group
from .. import h5, h5f, h5p, h5i, h5fd, _objects
from .. import version
def unregister_driver(name):
    """Unregister a custom driver.

    Parameters
    ----------
    name : str
        The name of the driver.
    """
    del _drivers[name]