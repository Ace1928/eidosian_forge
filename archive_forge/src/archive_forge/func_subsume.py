import errno
import itertools
import operator
import os
import stat
import sys
from bisect import bisect_left
from collections import deque
from io import BytesIO
import breezy
from .. import lazy_import
from . import bzrdir
import contextlib
from breezy import (
from breezy.bzr import (
from .. import errors, osutils
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from ..controldir import ControlDir
from ..lock import LogicalLockResult
from ..trace import mutter, note
from ..tree import (MissingNestedTree, TreeDirectory, TreeEntry, TreeFile,
from ..workingtree import WorkingTree, WorkingTreeFormat, format_registry
from .inventorytree import InventoryRevisionTree, MutableInventoryTree
def subsume(self, other_tree):

    def add_children(inventory, entry):
        for child_entry in entry.children.values():
            inventory._byid[child_entry.file_id] = child_entry
            if child_entry.kind == 'directory':
                add_children(inventory, child_entry)
    with self.lock_write():
        if other_tree.path2id('') == self.path2id(''):
            raise errors.BadSubsumeSource(self, other_tree, 'Trees have the same root')
        try:
            other_tree_path = self.relpath(other_tree.basedir)
        except errors.PathNotChild:
            raise errors.BadSubsumeSource(self, other_tree, 'Tree is not contained by the other')
        new_root_parent = self.path2id(osutils.dirname(other_tree_path))
        if new_root_parent is None:
            raise errors.BadSubsumeSource(self, other_tree, 'Parent directory is not versioned.')
        if not self.branch.repository.supports_rich_root():
            raise errors.SubsumeTargetNeedsUpgrade(other_tree)
        with other_tree.lock_tree_write():
            other_root = other_tree.root_inventory.root
            other_root.parent_id = new_root_parent
            other_root.name = osutils.basename(other_tree_path)
            self.root_inventory.add(other_root)
            add_children(self.root_inventory, other_root)
            self._write_inventory(self.root_inventory)
            for parent_id in other_tree.get_parent_ids():
                self.branch.fetch(other_tree.branch, parent_id)
                self.add_parent_tree_id(parent_id)
        other_tree.controldir.retire_bzrdir()