import os
from io import BytesIO
import breezy
from .. import config, controldir, errors, trace
from .. import transport as _mod_transport
from ..branch import Branch
from ..bzr.bzrdir import BzrDirMetaFormat1
from ..commit import (CannotCommitSelectedFileMerge, Commit,
from ..errors import BzrError, LockContention
from ..tree import TreeChange
from . import TestCase, TestCaseWithTransport, test_foreign
from .features import SymlinkFeature
from .matchers import MatchesAncestry, MatchesTreeChanges
def test_commit_new_subdir_child_selective(self):
    wt = self.make_branch_and_tree('.')
    b = wt.branch
    self.build_tree(['dir/', 'dir/file1', 'dir/file2'])
    wt.add(['dir', 'dir/file1', 'dir/file2'], ids=[b'dirid', b'file1id', b'file2id'])
    wt.commit('dir/file1', specific_files=['dir/file1'], rev_id=b'1')
    inv = b.repository.get_inventory(b'1')
    self.assertEqual(b'1', inv.get_entry(b'dirid').revision)
    self.assertEqual(b'1', inv.get_entry(b'file1id').revision)
    self.assertRaises(BzrError, inv.get_entry, b'file2id')