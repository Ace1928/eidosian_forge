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
def test_add_files(self):
    branch, tt = self.get_branch_and_transform()
    tt.new_file('file', tt.root, [b'contents'], b'file-id')
    trans_id = tt.new_directory('dir', tt.root, b'dir-id')
    if SymlinkFeature(self.test_dir).available():
        tt.new_symlink('symlink', trans_id, 'target', b'symlink-id')
    tt.commit(branch, 'message')
    tree = branch.basis_tree()
    self.assertEqual('file', tree.id2path(b'file-id'))
    self.assertEqual(b'contents', tree.get_file_text('file'))
    self.assertEqual('dir', tree.id2path(b'dir-id'))
    if SymlinkFeature(self.test_dir).available():
        self.assertEqual('dir/symlink', tree.id2path(b'symlink-id'))
        self.assertEqual('target', tree.get_symlink_target('dir/symlink'))