import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
def get_user_option_as_bool(self, option_name, expand=None, default=None):
    """Get a generic option as a boolean.

        Args:
          expand: Allow expanding references to other config values.
          default: Default value if nothing is configured

        Returns:
          None if the option doesn't exist or its value can't be
            interpreted as a boolean. Returns True or False otherwise.
        """
    s = self.get_user_option(option_name, expand=expand)
    if s is None:
        return default
    val = ui.bool_from_string(s)
    if val is None:
        trace.warning('Value "%s" is not a boolean for "%s"', s, option_name)
    return val