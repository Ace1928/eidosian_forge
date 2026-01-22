import os
import sys
from .lazy_import import lazy_import
from breezy import (
from . import errors
def config_path():
    """Return per-user configuration ini file filename."""
    path, kind = _config_dir()
    if kind == 'bazaar':
        return osutils.pathjoin(path, 'bazaar.conf')
    else:
        return osutils.pathjoin(path, 'breezy.conf')