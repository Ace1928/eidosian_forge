from both of those two places to another location.
import errno
import logging
import os
import sys
import time
from io import StringIO
import breezy
from .lazy_import import lazy_import
from breezy import (
from . import errors
def get_verbosity_level():
    """Get the verbosity level.

    See set_verbosity_level() for values.
    """
    return _verbosity_level