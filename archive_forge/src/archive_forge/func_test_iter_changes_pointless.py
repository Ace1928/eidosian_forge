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
def test_iter_changes_pointless(self):
    """Ensure that no-ops are not treated as modifications"""
    transform, root = self.transform()
    transform.new_file('old', root, [b'blah'], b'id-1')
    transform.new_directory('subdir', root, b'subdir-id')
    transform.apply()
    transform, root = self.transform()
    try:
        old = transform.trans_id_tree_path('old')
        subdir = transform.trans_id_tree_path('subdir')
        self.assertTreeChanges(transform, [])
        transform.delete_contents(subdir)
        transform.create_directory(subdir)
        transform.set_executability(False, old)
        transform.unversion_file(old)
        transform.version_file(old, file_id=b'id-1')
        transform.adjust_path('old', root, old)
        self.assertTreeChanges(transform, [])
    finally:
        transform.finalize()