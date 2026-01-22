import sys
import os
from warnings import warn
from .compat import filename_decode, filename_encode
from .base import phil, with_phil
from .group import Group
from .. import h5, h5f, h5p, h5i, h5fd, _objects
from .. import version
def register_driver(name, set_fapl):
    """Register a custom driver.

    Parameters
    ----------
    name : str
        The name of the driver.
    set_fapl : callable[PropFAID, **kwargs] -> NoneType
        The function to set the fapl to use your custom driver.
    """
    _drivers[name] = set_fapl