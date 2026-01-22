from breezy import controldir
from breezy.tests import TestCaseWithTransport
def test_uses_basis_tree_location(self):
    tree = self.make_tree_with_reference()
    tree.controldir.destroy_workingtree()
    out, err = self.run_bzr('reference', working_dir='tree')
    self.assertContainsRe(out, 'newpath http://example.org\n')