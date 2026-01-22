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
def fields_to_entry_0_parents(fields, _int=int):
    path_name_file_id_key = (fields[0], fields[1], fields[2])
    return (path_name_file_id_key, [(fields[3], fields[4], _int(fields[5]), fields[6] == b'y', fields[7])])