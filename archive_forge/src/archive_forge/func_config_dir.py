import os
import sys
from .lazy_import import lazy_import
from breezy import (
from . import errors
def config_dir():
    """Return per-user configuration directory as unicode string

    By default this is %APPDATA%/breezy on Windows, $XDG_CONFIG_HOME/breezy on
    Mac OS X and Linux. If the breezy config directory doesn't exist but
    the bazaar one (see bazaar_config_dir()) does, use that instead.
    """
    return _config_dir()[0]