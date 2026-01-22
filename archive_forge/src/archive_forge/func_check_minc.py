import os
import os.path
import warnings
from ..base import CommandLine
def check_minc():
    """Returns True if and only if MINC is installed.'"""
    return Info.version() is not None