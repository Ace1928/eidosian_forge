import codecs
import errno
import os
import sys
import time
from io import BytesIO, StringIO
import fastbencode as bencode
from .. import filters, osutils
from .. import revision as _mod_revision
from .. import rules, tests, trace, transform, urlutils
from ..bzr import generate_ids
from ..bzr.conflicts import (DeletingParent, DuplicateEntry, DuplicateID,
from ..controldir import ControlDir
from ..diff import show_diff_trees
from ..errors import (DuplicateKey, ExistingLimbo, ExistingPendingDeletion,
from ..merge import Merge3Merger, Merger
from ..mutabletree import MutableTree
from ..osutils import file_kind, pathjoin
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..transport import FileExists
from . import TestCaseInTempDir, TestSkipped, features
from .features import HardlinkFeature, SymlinkFeature
def modification_records(self):
    attribs = self.default_attribs()
    attribs[b'_id_number'] = 2
    attribs[b'_tree_path_ids'] = {b'file': b'new-1', b'': b'new-0'}
    attribs[b'_removed_contents'] = [b'new-1']
    contents = [(b'new-1', b'file', b'i 1\nz\n\nc 0 1 1 1\ni 1\nx\n\nc 0 3 3 1\n')]
    return self.make_records(attribs, contents)