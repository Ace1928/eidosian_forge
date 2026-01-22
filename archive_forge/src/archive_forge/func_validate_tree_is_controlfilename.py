from breezy.osutils import basename
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def validate_tree_is_controlfilename(self, tree):
    """check that 'tree' obeys the contract for is_control_filename."""
    bzrdirname = basename(tree.controldir.transport.base[:-1])
    self.assertTrue(tree.is_control_filename(bzrdirname))
    self.assertTrue(tree.is_control_filename(bzrdirname + '/subdir'))
    self.assertFalse(tree.is_control_filename('dir/' + bzrdirname))
    self.assertFalse(tree.is_control_filename('dir/' + bzrdirname + '/sub'))