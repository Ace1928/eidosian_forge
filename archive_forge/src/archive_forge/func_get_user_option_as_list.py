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
def get_user_option_as_list(self, option_name, expand=None):
    """Get a generic option as a list - no special process, no default.

        Returns:
          None if the option doesn't exist. Returns the value as a list
          otherwise.
        """
    l = self.get_user_option(option_name, expand=expand)
    if isinstance(l, str):
        l = [l]
    return l