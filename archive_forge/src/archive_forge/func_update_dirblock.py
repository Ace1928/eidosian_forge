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
def update_dirblock(from_dir, to_key, to_dir_utf8):
    """Recursively update all entries in this dirblock."""
    if from_dir == b'':
        raise AssertionError('renaming root not supported')
    from_key = (from_dir, '')
    from_block_idx, present = state._find_block_index_from_key(from_key)
    if not present:
        return
    from_block = state._dirblocks[from_block_idx]
    to_block_index, to_entry_index, _, _ = state._get_block_entry_index(to_key[0], to_key[1], 0)
    to_block_index = state._ensure_block(to_block_index, to_entry_index, to_dir_utf8)
    to_block = state._dirblocks[to_block_index]
    for entry in from_block[1][:]:
        if not entry[0][0] == from_dir:
            raise AssertionError()
        cur_details = entry[1][0]
        to_key = (to_dir_utf8, entry[0][1], entry[0][2])
        from_path_utf8 = osutils.pathjoin(entry[0][0], entry[0][1])
        to_path_utf8 = osutils.pathjoin(to_dir_utf8, entry[0][1])
        minikind = cur_details[0]
        if minikind in (b'a', b'r'):
            continue
        move_one(entry, from_path_utf8=from_path_utf8, minikind=minikind, executable=cur_details[3], fingerprint=cur_details[1], packed_stat=cur_details[4], size=cur_details[2], to_block=to_block, to_key=to_key, to_path_utf8=to_path_utf8)
        if minikind == b'd':
            update_dirblock(from_path_utf8, to_key, to_path_utf8)