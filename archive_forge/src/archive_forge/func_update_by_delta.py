import bisect
import codecs
import contextlib
import errno
import operator
import os
import stat
import sys
import time
import zlib
from stat import S_IEXEC
from .. import (cache_utf8, config, debug, errors, lock, osutils, trace,
from . import inventory, static_tuple
from .inventorytree import InventoryTreeChange
def update_by_delta(self, delta):
    """Apply an inventory delta to the dirstate for tree 0

        This is the workhorse for apply_inventory_delta in dirstate based
        trees.

        :param delta: An inventory delta.  See Inventory.apply_delta for
            details.
        """
    self._read_dirblocks_if_needed()
    encode = cache_utf8.encode
    insertions = {}
    removals = {}
    parents = set()
    new_ids = set()
    delta = self._check_delta_is_valid(delta)
    for old_path, new_path, file_id, inv_entry in delta:
        if not isinstance(file_id, bytes):
            raise AssertionError('must be a utf8 file_id not {}'.format(type(file_id)))
        if file_id in insertions or file_id in removals:
            self._raise_invalid(old_path or new_path, file_id, 'repeated file_id')
        if old_path is not None:
            old_path = old_path.encode('utf-8')
            removals[file_id] = old_path
        else:
            new_ids.add(file_id)
        if new_path is not None:
            if inv_entry is None:
                self._raise_invalid(new_path, file_id, 'new_path with no entry')
            new_path = new_path.encode('utf-8')
            dirname_utf8, basename = osutils.split(new_path)
            if basename:
                parents.add((dirname_utf8, inv_entry.parent_id))
            key = (dirname_utf8, basename, file_id)
            minikind = DirState._kind_to_minikind[inv_entry.kind]
            if minikind == b't':
                fingerprint = inv_entry.reference_revision or b''
            else:
                fingerprint = b''
            insertions[file_id] = (key, minikind, inv_entry.executable, fingerprint, new_path)
        if None not in (old_path, new_path):
            for child in self._iter_child_entries(0, old_path):
                if child[0][2] in insertions or child[0][2] in removals:
                    continue
                child_dirname = child[0][0]
                child_basename = child[0][1]
                minikind = child[1][0][0]
                fingerprint = child[1][0][4]
                executable = child[1][0][3]
                old_child_path = osutils.pathjoin(child_dirname, child_basename)
                removals[child[0][2]] = old_child_path
                child_suffix = child_dirname[len(old_path):]
                new_child_dirname = new_path + child_suffix
                key = (new_child_dirname, child_basename, child[0][2])
                new_child_path = osutils.pathjoin(new_child_dirname, child_basename)
                insertions[child[0][2]] = (key, minikind, executable, fingerprint, new_child_path)
    self._check_delta_ids_absent(new_ids, delta, 0)
    try:
        self._apply_removals(removals.items())
        self._apply_insertions(insertions.values())
        self._after_delta_check_parents(parents, 0)
    except errors.BzrError as e:
        self._changes_aborted = True
        if 'integrity error' not in str(e):
            raise
        raise errors.InconsistentDeltaDelta(delta, 'error from _get_entry. {}'.format(e))