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
def from_unicode(self, unicode_str):
    if not isinstance(unicode_str, str):
        raise TypeError
    try:
        return self.registry.get(unicode_str)
    except KeyError:
        raise ValueError('Invalid value %s for %s.See help for a list of possible values.' % (unicode_str, self.name))