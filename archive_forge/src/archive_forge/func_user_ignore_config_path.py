import os
import sys
from .lazy_import import lazy_import
from breezy import (
from . import errors
def user_ignore_config_path():
    """Return per-user authentication ini file filename."""
    return osutils.pathjoin(config_dir(), 'ignore')