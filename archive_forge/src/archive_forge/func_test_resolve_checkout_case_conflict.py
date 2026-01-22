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
def test_resolve_checkout_case_conflict(self):
    tree = self.make_branch_and_tree('tree')
    tree.case_sensitive = False
    transform = tree.transform()
    self.addCleanup(transform.finalize)
    transform.new_file('file', transform.root, [b'content'])
    transform.new_file('FiLe', transform.root, [b'content'])
    resolve_conflicts(transform, pass_func=lambda t, c: resolve_checkout(t, c, []))
    transform.apply()
    self.assertPathExists('tree/file')
    self.assertPathExists('tree/FiLe.moved')