import os
from io import BytesIO
from ..lazy_import import lazy_import
import contextlib
import errno
import stat
from breezy import (
from breezy.bzr import (
from .. import errors
from .. import revision as _mod_revision
from ..lock import LogicalLockResult
from ..lockable_files import LockableFiles
from ..lockdir import LockDir
from ..mutabletree import BadReferenceTarget, MutableTree
from ..osutils import file_kind, isdir, pathjoin, realpath, safe_unicode
from ..transport import NoSuchFile, get_transport_from_path
from ..transport.local import LocalTransport
from ..tree import FileTimestampUnavailable, InterTree, MissingNestedTree
from ..workingtree import WorkingTree
from . import dirstate
from .inventory import ROOT_ID, Inventory, entry_factory
from .inventorytree import (InterInventoryTree, InventoryRevisionTree,
from .workingtree import InventoryWorkingTree, WorkingTreeFormatMetaDir
def paths2ids(self, paths, trees=[], require_versioned=True):
    """See Tree.paths2ids().

        This specialisation fast-paths the case where all the trees are in the
        dirstate.
        """
    if paths is None:
        return None
    parents = self.get_parent_ids()
    for tree in trees:
        if not (isinstance(tree, DirStateRevisionTree) and tree._revision_id in parents):
            return super().paths2ids(paths, trees, require_versioned)
    search_indexes = [0] + [1 + parents.index(tree._revision_id) for tree in trees]
    paths_utf8 = set()
    for path in paths:
        paths_utf8.add(path.encode('utf8'))
    state = self.current_dirstate()
    if False and (state._dirblock_state == dirstate.DirState.NOT_IN_MEMORY and b'' not in paths):
        paths2ids = self._paths2ids_using_bisect
    else:
        paths2ids = self._paths2ids_in_memory
    return paths2ids(paths_utf8, search_indexes, require_versioned=require_versioned)