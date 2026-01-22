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
def iter_option_refs(string):
    is_ref = False
    for chunk in _option_ref_re.split(string):
        yield (is_ref, chunk)
        is_ref = not is_ref