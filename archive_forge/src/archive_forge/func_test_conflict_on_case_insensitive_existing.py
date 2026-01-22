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
def test_conflict_on_case_insensitive_existing(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/FiLe'])
    tree.case_sensitive = True
    transform = tree.transform()
    self.addCleanup(transform.finalize)
    transform.new_file('file', transform.root, [b'content'])
    result = transform.find_raw_conflicts()
    self.assertEqual([], result)
    transform.finalize()
    tree.case_sensitive = False
    transform = tree.transform()
    self.addCleanup(transform.finalize)
    transform.new_file('file', transform.root, [b'content'])
    result = transform.find_raw_conflicts()
    self.assertEqual([('duplicate', 'new-1', 'new-2', 'file')], result)