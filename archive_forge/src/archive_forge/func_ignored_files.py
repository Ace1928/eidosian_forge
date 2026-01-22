import contextlib
import errno
import os
import sys
from typing import TYPE_CHECKING, Optional, Tuple
import breezy
from .lazy_import import lazy_import
import stat
from breezy import (
from . import errors, mutabletree, osutils
from . import revision as _mod_revision
from .controldir import (ControlComponent, ControlComponentFormat,
from .i18n import gettext
from .symbol_versioning import deprecated_in, deprecated_method
from .trace import mutter, note
from .transport import NoSuchFile
def ignored_files(self):
    """Yield list of PATH, IGNORE_PATTERN"""
    for subp in self.extras():
        pat = self.is_ignored(subp)
        if pat is not None:
            yield (subp, pat)