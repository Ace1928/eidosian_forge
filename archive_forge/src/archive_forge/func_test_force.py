from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
def test_force(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/file'])
    tree.add('file')
    self.run_bzr_error(['Working tree ".*" has uncommitted changes'], 'reconfigure --branch tree')
    self.run_bzr('reconfigure --force --branch tree')