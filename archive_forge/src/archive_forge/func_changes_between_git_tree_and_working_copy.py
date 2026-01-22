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
def changes_between_git_tree_and_working_copy(source_store, from_tree_sha, target, want_unchanged=False, want_unversioned=False, rename_detector=None, include_trees=True):
    """Determine the changes between a git tree and a working tree with index.

    """
    to_tree_sha, extras = target.git_snapshot(want_unversioned=want_unversioned)
    store = OverlayObjectStore([source_store, target.store])
    return (tree_changes(store, from_tree_sha, to_tree_sha, include_trees=include_trees, rename_detector=rename_detector, want_unchanged=want_unchanged, change_type_same=True), extras)