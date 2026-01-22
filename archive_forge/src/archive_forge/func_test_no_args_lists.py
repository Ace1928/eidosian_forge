from breezy import controldir
from breezy.tests import TestCaseWithTransport
def test_no_args_lists(self):
    tree = self.make_branch_and_tree('branch')
    branch = tree.branch
    tree.add_reference(self.make_branch_and_tree('branch/path'))
    tree.add_reference(self.make_branch_and_tree('branch/lath'))
    tree.set_reference_info('path', 'http://example.org')
    tree.set_reference_info('lath', 'http://example.org/2')
    out, err = self.run_bzr('reference', working_dir='branch')
    lines = out.splitlines()
    self.assertEqual('lath http://example.org/2', lines[0])
    self.assertEqual('path http://example.org', lines[1])