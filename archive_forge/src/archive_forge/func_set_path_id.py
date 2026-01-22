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
def set_path_id(self, path, new_id):
    """Change the id of path to new_id in the current working tree.

        :param path: The path inside the tree to set - b'' is the root, 'foo'
            is the path foo in the root.
        :param new_id: The new id to assign to the path. This must be a utf8
            file id (not unicode, and not None).
        """
    self._read_dirblocks_if_needed()
    if len(path):
        raise NotImplementedError(self.set_path_id)
    entry = self._get_entry(0, path_utf8=path)
    if entry[0][2] == new_id:
        return
    if new_id.__class__ != bytes:
        raise AssertionError('must be a utf8 file_id not {}'.format(type(new_id)))
    self._make_absent(entry)
    self.update_minimal((b'', b'', new_id), b'd', path_utf8=b'', packed_stat=entry[1][0][4])
    self._mark_modified()