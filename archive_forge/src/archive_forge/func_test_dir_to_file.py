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
def test_dir_to_file(self):
    wt = self.make_branch_and_tree('.')
    self.build_tree(['foo/', 'foo/bar'])
    wt.add(['foo', 'foo/bar'])
    wt.commit('one')
    tt = wt.transform()
    self.addCleanup(tt.finalize)
    foo_trans_id = tt.trans_id_tree_path('foo')
    bar_trans_id = tt.trans_id_tree_path('foo/bar')
    tt.delete_contents(foo_trans_id)
    tt.delete_versioned(bar_trans_id)
    tt.create_file([b'aa\n'], foo_trans_id)
    tt.apply()
    self.assertPathExists('foo')
    wt.lock_read()
    self.addCleanup(wt.unlock)
    self.assertEqual(wt.kind('foo'), 'file')