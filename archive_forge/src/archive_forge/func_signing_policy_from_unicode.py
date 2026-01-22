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
def signing_policy_from_unicode(signature_string):
    """Convert a string to a signing policy."""
    if signature_string.lower() == 'when-required':
        return SIGN_WHEN_REQUIRED
    if signature_string.lower() == 'never':
        return SIGN_NEVER
    if signature_string.lower() == 'always':
        return SIGN_ALWAYS
    if signature_string.lower() == 'when-possible':
        return SIGN_WHEN_POSSIBLE
    raise ValueError("Invalid signing policy '%s'" % signature_string)