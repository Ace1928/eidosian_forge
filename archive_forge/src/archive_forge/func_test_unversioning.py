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
def test_unversioning(self):
    create_tree, root = self.transform()
    parent_id = create_tree.new_directory('parent', root, b'parent-id')
    create_tree.new_file('child', parent_id, [b'child'], b'child-id')
    create_tree.apply()
    unversion = self.wt.transform()
    self.addCleanup(unversion.finalize)
    parent = unversion.trans_id_tree_path('parent')
    unversion.unversion_file(parent)
    if self.wt.has_versioned_directories():
        self.assertEqual(unversion.find_raw_conflicts(), [('unversioned parent', parent_id)])
    else:
        self.assertEqual(unversion.find_raw_conflicts(), [])
    file_id = unversion.trans_id_tree_path('parent/child')
    unversion.unversion_file(file_id)
    unversion.apply()