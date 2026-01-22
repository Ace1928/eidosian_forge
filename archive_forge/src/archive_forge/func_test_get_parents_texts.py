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
def test_get_parents_texts(self):
    LINES_ONE = b'aa\nbb\ncc\ndd\n'
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/file', LINES_ONE)])
    tree.add('file', ids=b'file-id')
    tt = self.get_preview(tree)
    trans_id = tt.trans_id_tree_path('file')
    self.assertEqual((LINES_ONE,), tt._get_parents_texts(trans_id))