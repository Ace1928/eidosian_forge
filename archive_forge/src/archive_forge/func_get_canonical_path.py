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
def get_canonical_path(self, path):
    """Returns the first item in the tree that matches a path.

        This is meant to allow case-insensitive path lookups on e.g.
        FAT filesystems.

        If a path matches exactly, it is returned. If no path matches exactly
        but more than one path matches according to the underlying file system,
        it is implementation defined which is returned.

        If no path matches according to the file system, the input path is
        returned, but with as many path entries that do exist changed to their
        canonical form.

        If you need to resolve many names from the same tree, you should
        use get_canonical_paths() to avoid O(N) behaviour.

        Args:
          path: A paths relative to the root of the tree.

        Returns:
          The input path adjusted to account for existing elements
          that match case insensitively.
        """
    with self.lock_read():
        return next(self.get_canonical_paths([path]))