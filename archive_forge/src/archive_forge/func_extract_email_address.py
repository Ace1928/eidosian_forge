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
def extract_email_address(e):
    """Return just the address part of an email string.

    That is just the user@domain part, nothing else.
    This part is required to contain only ascii characters.
    If it can't be extracted, raises an error.

    >>> extract_email_address('Jane Tester <jane@test.com>')
    "jane@test.com"
    """
    name, email = parse_username(e)
    if not email:
        raise NoEmailInUsername(e)
    return email