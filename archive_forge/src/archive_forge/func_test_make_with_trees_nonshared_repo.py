from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
def test_make_with_trees_nonshared_repo(self):
    branch = self.make_branch('branch')
    self.run_bzr_error(["Requested reconfiguration of '.*' is not supported"], 'reconfigure --with-trees branch')