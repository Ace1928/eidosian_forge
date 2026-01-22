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
def transform_tree(from_tree, to_tree, interesting_files=None):
    with from_tree.lock_tree_write():
        merge_inner(from_tree.branch, to_tree, from_tree, ignore_zero=True, this_tree=from_tree, interesting_files=interesting_files)