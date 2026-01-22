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
def signature_policy_from_unicode(signature_string):
    """Convert a string to a signing policy."""
    if signature_string.lower() == 'check-available':
        return CHECK_IF_POSSIBLE
    if signature_string.lower() == 'ignore':
        return CHECK_NEVER
    if signature_string.lower() == 'require':
        return CHECK_ALWAYS
    raise ValueError("Invalid signatures policy '%s'" % signature_string)