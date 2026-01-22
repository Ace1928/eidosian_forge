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
def int_SI_from_store(unicode_str):
    """Convert a human readable size in SI units, e.g 10MB into an integer.

    Accepted suffixes are K,M,G. It is case-insensitive and may be followed
    by a trailing b (i.e. Kb, MB). This is intended to be practical and not
    pedantic.

    Returns: Integer, expanded to its base-10 value if a proper SI unit is
        found, None otherwise.
    """
    regexp = '^(\\d+)(([' + ''.join(_unit_suffixes) + '])b?)?$'
    p = re.compile(regexp, re.IGNORECASE)
    m = p.match(unicode_str)
    val = None
    if m is not None:
        val, _, unit = m.groups()
        val = int(val)
        if unit:
            try:
                coeff = _unit_suffixes[unit.upper()]
            except KeyError:
                raise ValueError(gettext('{0} is not an SI unit.').format(unit))
            val *= coeff
    return val