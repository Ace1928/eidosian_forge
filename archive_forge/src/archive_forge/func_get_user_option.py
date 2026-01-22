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
def get_user_option(self, option_name, expand=True):
    """Get a generic option - no special process, no default.

        Args:
          option_name: The queried option.
          expand: Whether options references should be expanded.

        Returns:
          The value of the option.
        """
    value = self._get_user_option(option_name)
    if expand:
        if isinstance(value, list):
            value = self._expand_options_in_list(value)
        elif isinstance(value, dict):
            trace.warning('Cannot expand "%s": Dicts do not support option expansion' % (option_name,))
        else:
            value = self._expand_options_in_string(value)
    for hook in OldConfigHooks['get']:
        hook(self, option_name, value)
    return value