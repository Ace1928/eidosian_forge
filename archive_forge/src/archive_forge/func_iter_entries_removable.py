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
def iter_entries_removable():
    for block in self._dirblocks:
        deleted_positions = []
        for pos, entry in enumerate(block[1]):
            yield entry
            if (entry[1][0][0], entry[1][1][0]) in dead_patterns:
                deleted_positions.append(pos)
        if deleted_positions:
            if len(deleted_positions) == len(block[1]):
                del block[1][:]
            else:
                for pos in reversed(deleted_positions):
                    del block[1][pos]