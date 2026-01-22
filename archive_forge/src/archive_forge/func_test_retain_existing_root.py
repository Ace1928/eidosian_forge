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
def test_retain_existing_root(self):
    tt, root = self.transform()
    with tt:
        tt.new_directory('', ROOT_PARENT, b'new-root-id')
        tt.fixup_new_roots()
        if self.wt.has_versioned_directories():
            self.assertTrue(tt.final_is_versioned(tt.root))
        if self.wt.supports_setting_file_ids():
            self.assertNotEqual(b'new-root-id', tt.final_file_id(tt.root))