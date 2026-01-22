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
def test_two_directories_clash(self):

    def tt_helper():
        wt = self.make_branch_and_tree('.')
        tt = wt.transform()
        try:
            foo_1 = tt.new_directory('foo', tt.root)
            tt.new_directory('bar', foo_1)
            foo_2 = tt.new_directory('foo', tt.root)
            tt.new_directory('baz', foo_2)
            tt.apply(no_conflicts=True)
        except BaseException:
            wt.unlock()
            raise
    err = self.assertRaises(FileExists, tt_helper)
    self.assertEndsWith(err.path, '/foo')