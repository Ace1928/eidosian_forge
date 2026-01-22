import contextlib
import errno
import os
import tempfile
import time
from stat import S_IEXEC, S_ISREG
from .. import (annotate, conflicts, controldir, errors, lock, multiparent,
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from .. import tree, ui, urlutils
from ..filters import ContentFilterContext, filtered_output_bytes
from ..i18n import gettext
from ..mutabletree import MutableTree
from ..progress import ProgressPhase
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..tree import find_previous_path
from . import inventory, inventorytree
from .conflicts import Conflict
def build_tree(tree, wt, accelerator_tree=None, hardlink=False, delta_from_tree=False):
    """Create working tree for a branch, using a TreeTransform.

    This function should be used on empty trees, having a tree root at most.
    (see merge and revert functionality for working with existing trees)

    Existing files are handled like so:

    - Existing bzrdirs take precedence over creating new items.  They are
      created as '%s.diverted' % name.
    - Otherwise, if the content on disk matches the content we are building,
      it is silently replaced.
    - Otherwise, conflict resolution will move the old file to 'oldname.moved'.

    :param tree: The tree to convert wt into a copy of
    :param wt: The working tree that files will be placed into
    :param accelerator_tree: A tree which can be used for retrieving file
        contents more quickly than tree itself, i.e. a workingtree.  tree
        will be used for cases where accelerator_tree's content is different.
    :param hardlink: If true, hard-link files to accelerator_tree, where
        possible.  accelerator_tree must implement abspath, i.e. be a
        working tree.
    :param delta_from_tree: If true, build_tree may use the input Tree to
        generate the inventory delta.
    """
    with contextlib.ExitStack() as exit_stack:
        exit_stack.enter_context(wt.lock_tree_write())
        exit_stack.enter_context(tree.lock_read())
        if accelerator_tree is not None:
            exit_stack.enter_context(accelerator_tree.lock_read())
        return _build_tree(tree, wt, accelerator_tree, hardlink, delta_from_tree)