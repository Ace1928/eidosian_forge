import os
import os.path
import warnings
from ..base import CommandLine
def no_minc():
    """Returns True if and only if MINC is *not* installed."""
    return not check_minc()