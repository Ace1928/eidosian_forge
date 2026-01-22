import os
import time
from ... import errors, osutils
from ...lockdir import LockDir
from ...tests import TestCaseWithTransport, TestSkipped, features
from ...tree import InterTree
from .. import bzrdir, dirstate, inventory, workingtree_4
def test_get_file_with_stat_id_only(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['foo'])
    tree.add(['foo'])
    tree.lock_read()
    self.addCleanup(tree.unlock)
    file_obj, statvalue = tree.get_file_with_stat('foo')
    expected = os.lstat('foo')
    self.assertEqualStat(expected, statvalue)
    self.assertEqual([b'contents of foo\n'], file_obj.readlines())