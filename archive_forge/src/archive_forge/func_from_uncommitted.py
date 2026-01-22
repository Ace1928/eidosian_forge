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
def from_uncommitted(tree, other_tree, base_tree=None):
    """Return a Merger for uncommitted changes in other_tree.

        :param tree: The tree to merge into
        :param other_tree: The tree to get uncommitted changes from
        :param base_tree: The basis to use for the merge.  If unspecified,
            other_tree.basis_tree() will be used.
        """
    if base_tree is None:
        base_tree = other_tree.basis_tree()
    merger = Merger(tree.branch, other_tree, base_tree, tree)
    merger.base_rev_id = merger.base_tree.get_revision_id()
    merger.other_rev_id = None
    merger.other_basis = merger.base_rev_id
    return merger