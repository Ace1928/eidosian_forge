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
def test_resolve_conflicts_missing_parent(self):
    wt = self.make_branch_and_tree('.')
    tt = wt.transform()
    self.addCleanup(tt.finalize)
    parent = tt.assign_id()
    tt.new_file('file', parent, [b'Contents'])
    raw_conflicts = resolve_conflicts(tt)
    self.assertLength(1, raw_conflicts)
    self.assertEqual(('missing parent', 'Created directory', 'new-1'), raw_conflicts.pop())
    self.assertRaises(NoFinalPath, tt.apply)