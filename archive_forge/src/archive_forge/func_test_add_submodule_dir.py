import os
import stat
from dulwich import __version__ as dulwich_version
from dulwich.diff_tree import RenameDetector, tree_changes
from dulwich.index import IndexEntry, ConflictedIndexEntry
from dulwich.object_store import OverlayObjectStore
from dulwich.objects import S_IFGITLINK, ZERO_SHA, Blob, Tree
from ... import conflicts as _mod_conflicts
from ... import workingtree as _mod_workingtree
from ...bzr.inventorytree import InventoryTreeChange as TreeChange
from ...delta import TreeDelta
from ...tests import TestCase, TestCaseWithTransport
from ..mapping import default_mapping
from ..tree import tree_delta_from_git_changes
def test_add_submodule_dir(self):
    subtree = self.make_branch_and_tree('asub', format='git')
    subtree.commit('Empty commit')
    self.tree.add(['asub'])
    with self.tree.lock_read():
        entry = self.tree.index[b'asub']
        self.assertEqual(entry.mode, S_IFGITLINK)
    self.assertEqual([], list(subtree.unknowns()))