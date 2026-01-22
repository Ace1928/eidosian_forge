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
def test_hardlink(self):
    self.requireFeature(HardlinkFeature(self.test_dir))
    transform, root = self.transform()
    transform.new_file('file1', root, [b'contents'])
    transform.apply()
    target = self.make_branch_and_tree('target')
    target_transform = target.transform()
    trans_id = target_transform.create_path('file1', target_transform.root)
    target_transform.create_hardlink(self.wt.abspath('file1'), trans_id)
    target_transform.apply()
    self.assertPathExists('target/file1')
    source_stat = os.stat(self.wt.abspath('file1'))
    target_stat = os.stat('target/file1')
    self.assertEqual(source_stat, target_stat)