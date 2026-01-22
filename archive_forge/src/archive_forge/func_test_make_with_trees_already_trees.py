from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
def test_make_with_trees_already_trees(self):
    repo = self.make_repository('repo', shared=True)
    repo.set_make_working_trees(True)
    self.run_bzr_error([' already creates working trees'], 'reconfigure --with-trees repo')