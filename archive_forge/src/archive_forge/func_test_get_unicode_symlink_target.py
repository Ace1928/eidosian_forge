import os
from breezy import osutils, tests
from breezy.tests import features, per_tree
def test_get_unicode_symlink_target(self):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    self.requireFeature(features.UnicodeFilenameFeature)
    tree = self.make_branch_and_tree('tree')
    target = 'targ€t'
    os.symlink(target, os.fsencode('tree/β_link'))
    tree.add(['β_link'])
    tree.lock_read()
    self.addCleanup(tree.unlock)
    actual = tree.get_symlink_target('β_link')
    self.assertEqual(target, actual)