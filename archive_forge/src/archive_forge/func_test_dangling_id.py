import os
from breezy import tests
from breezy.bzr import inventory
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_dangling_id(self):
    wt = self.make_branch_and_tree('b1')
    wt.lock_tree_write()
    self.addCleanup(wt.unlock)
    self.assertEqual(len(list(wt.all_versioned_paths())), 1)
    with open('b1/a', 'wb') as f:
        f.write(b'a test\n')
    wt.add('a')
    self.assertEqual(len(list(wt.all_versioned_paths())), 2)
    wt.flush()
    os.unlink('b1/a')
    wt.revert()
    self.assertEqual(len(list(wt.all_versioned_paths())), 1)