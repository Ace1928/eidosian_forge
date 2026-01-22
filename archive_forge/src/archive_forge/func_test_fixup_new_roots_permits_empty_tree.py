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
def test_fixup_new_roots_permits_empty_tree(self):
    transform, root = self.transform()
    transform.delete_contents(root)
    transform.unversion_file(root)
    transform.fixup_new_roots()
    self.assertIs(None, transform.final_kind(root))
    if self.wt.supports_setting_file_ids():
        self.assertIs(None, transform.final_file_id(root))