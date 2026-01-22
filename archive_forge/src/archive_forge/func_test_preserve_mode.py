import errno
import os
import sys
import time
from io import BytesIO
from breezy.bzr.transform import resolve_checkout
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from ... import osutils, tests, trace, transform, urlutils
from ...bzr.conflicts import (DeletingParent, DuplicateEntry, DuplicateID,
from ...errors import (DuplicateKey, ExistingLimbo, ExistingPendingDeletion,
from ...osutils import file_kind, pathjoin
from ...transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ...transport import FileExists
from ...tree import TreeChange
from .. import TestSkipped, features
from ..features import HardlinkFeature, SymlinkFeature
def test_preserve_mode(self):
    """File mode is preserved when replacing content"""
    if sys.platform == 'win32':
        raise TestSkipped('chmod has no effect on win32')
    transform, root = self.transform()
    transform.new_file('file1', root, [b'contents'], b'file1-id', True)
    transform.apply()
    self.wt.lock_write()
    self.addCleanup(self.wt.unlock)
    self.assertTrue(self.wt.is_executable('file1'))
    transform, root = self.transform()
    file1_id = transform.trans_id_tree_path('file1')
    transform.delete_contents(file1_id)
    transform.create_file([b'contents2'], file1_id)
    transform.apply()
    self.assertTrue(self.wt.is_executable('file1'))