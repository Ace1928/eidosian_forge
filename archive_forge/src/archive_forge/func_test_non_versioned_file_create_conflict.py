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
def test_non_versioned_file_create_conflict(self):
    wt, tt = self.make_tt_with_versioned_dir()
    dir_tid = tt.trans_id_tree_path('dir')
    tt.new_file('file', dir_tid, [b'Contents'])
    tt.delete_contents(dir_tid)
    tt.unversion_file(dir_tid)
    conflicts = resolve_conflicts(tt)
    self.assertLength(1, conflicts)
    self.assertEqual(('deleting parent', 'Not deleting', 'new-1'), conflicts.pop())