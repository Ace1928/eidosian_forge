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
def test_replace_dangling_ie(self):
    create_tree, root = self.transform()
    root = create_tree.root
    create_tree.new_file('name1', root, [b'hello1'], b'name1')
    create_tree.apply()
    delete_contents = self.wt.transform()
    self.addCleanup(delete_contents.finalize)
    file = delete_contents.trans_id_tree_path('name1')
    delete_contents.delete_contents(file)
    delete_contents.apply()
    delete_contents.finalize()
    replace = self.wt.transform()
    self.addCleanup(replace.finalize)
    name2 = replace.new_file('name2', root, [b'hello2'], b'name1')
    conflicts = replace.find_raw_conflicts()
    name1 = replace.trans_id_tree_path('name1')
    if self.wt.supports_setting_file_ids():
        self.assertEqual(conflicts, [('duplicate id', name1, name2)])
    else:
        self.assertEqual(conflicts, [])
    resolve_conflicts(replace)
    replace.apply()