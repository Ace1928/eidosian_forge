import base64
import os
import pprint
from io import BytesIO
from ... import cache_utf8, osutils, timestamp
from ...errors import BzrError, NoSuchId, TestamentMismatch
from ...osutils import pathjoin, sha_string, sha_strings
from ...revision import NULL_REVISION, Revision
from ...trace import mutter, warning
from ...tree import InterTree, Tree
from ..inventory import (Inventory, InventoryDirectory, InventoryFile,
from ..inventorytree import InventoryTree
from ..testament import StrictTestament
from ..xml5 import serializer_v5
from . import apply_bundle
def renamed(kind, extra, lines):
    info = extra.split(' // ')
    if len(info) < 2:
        raise BzrError('renamed action lines need both a from and to: %r' % extra)
    old_path = info[0]
    if info[1].startswith('=> '):
        new_path = info[1][3:]
    else:
        new_path = info[1]
    bundle_tree.note_rename(old_path, new_path)
    last_modified, encoding = extra_info(info[2:], new_path)
    revision = get_rev_id(last_modified, new_path, kind)
    if lines:
        do_patch(new_path, lines, encoding)