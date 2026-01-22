from contextlib import contextmanager
import logging
import typing
import os
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import ffi_proxy
from rpy2.rinterface_lib import conversion
def processevents() -> None:
    """Process R events.

    This function can be periodically called by R to handle
    events such as window resizing in an interactive graphical
    device."""
    pass