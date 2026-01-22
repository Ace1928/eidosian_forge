import contextlib
import tempfile
from typing import Type
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from . import decorators, errors, hooks, osutils, registry
from . import revision as _mod_revision
from . import trace, transform
from . import transport as _mod_transport
from . import tree as _mod_tree
@staticmethod
def contents_sha1(tree, path):
    """Determine the sha1 of the file contents (used as a key method)."""
    try:
        return tree.get_file_sha1(path)
    except _mod_transport.NoSuchFile:
        return None