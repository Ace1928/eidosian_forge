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
def test_reuse_when_first_moved(self):
    """Don't avoid direct paths when it is safe to use them"""
    transform, root = self.transform()
    parent = transform.new_directory('parent', root)
    child1 = transform.new_directory('child', parent)
    transform.adjust_path('child1', parent, child1)
    transform.new_directory('child', parent)
    transform.apply()
    self.assertEqual(1, transform.rename_count)