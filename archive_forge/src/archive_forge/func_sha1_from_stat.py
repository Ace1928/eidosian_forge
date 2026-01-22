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
def sha1_from_stat(self, path, stat_result):
    """Find a sha1 given a stat lookup."""
    return self._get_packed_stat_index().get(pack_stat(stat_result), None)