import sys
from breezy import branch, errors
from breezy.tests import TestSkipped
from breezy.tests.matchers import *
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_unlock_from_tree_write_lock_flushes(self):
    self._test_unlock_with_lock_method('lock_tree_write')