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
def delta_key(d):
    old_path, new_path, file_id, new_entry = d
    if old_path is None:
        old_path = ''
    if new_path is None:
        new_path = ''
    return (old_path, new_path, file_id, new_entry)