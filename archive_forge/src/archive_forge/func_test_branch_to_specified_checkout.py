from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
def test_branch_to_specified_checkout(self):
    branch = self.make_branch('branch')
    parent = self.make_branch('parent')
    self.run_bzr('reconfigure branch --checkout --bind-to parent')