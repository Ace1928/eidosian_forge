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
def test_both_rename3(self):
    create_tree, root = self.transform()
    tests = create_tree.new_directory('tests', root, b'tests-id')
    create_tree.new_file('test_too_much.py', tests, [b'hello1'], b'test_too_much-id')
    create_tree.apply()
    mangle_tree, root = self.transform()
    tests = mangle_tree.trans_id_tree_path('tests')
    test_too_much = mangle_tree.trans_id_tree_path('tests/test_too_much.py')
    mangle_tree.adjust_path('selftest', root, tests)
    mangle_tree.adjust_path('blackbox.py', tests, test_too_much)
    mangle_tree.set_executability(True, test_too_much)
    mangle_tree.apply()