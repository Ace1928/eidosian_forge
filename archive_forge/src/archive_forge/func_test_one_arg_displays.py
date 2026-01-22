from breezy import controldir
from breezy.tests import TestCaseWithTransport
def test_one_arg_displays(self):
    tree = self.make_tree_with_reference()
    out, err = self.run_bzr('reference newpath', working_dir='tree')
    self.assertEqual('newpath http://example.org\n', out)