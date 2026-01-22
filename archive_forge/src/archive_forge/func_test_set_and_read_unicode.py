import sys
from breezy import errors
from breezy.tests import TestSkipped
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_set_and_read_unicode(self):
    if sys.platform == 'win32':
        raise TestSkipped("don't use oslocks on win32 in unix manner")
    self.thisFailsStrictLockCheck()
    tree = self.make_branch_and_tree('a-tree')
    if not tree.supports_setting_file_ids():
        self.skipTest('format does not support setting file ids')
    root_id = 'ån-id'.encode()
    with tree.lock_write():
        old_id = tree.path2id('')
        tree.set_root_id(root_id)
        self.assertEqual(root_id, tree.path2id(''))
        reference_tree = tree.controldir.open_workingtree()
        self.assertEqual(old_id, reference_tree.path2id(''))
    self.assertEqual(root_id, tree.path2id(''))
    tree = tree.controldir.open_workingtree()
    self.assertEqual(root_id, tree.path2id(''))
    tree._validate()