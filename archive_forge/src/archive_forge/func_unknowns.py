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
def unknowns(self):
    """Return all unknown files.

        These are files in the working directory that are not versioned or
        control files or ignored.
        """
    with self.lock_read():
        return iter([subp for subp in self.extras() if not self.is_ignored(subp)])