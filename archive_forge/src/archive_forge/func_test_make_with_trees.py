from breezy import controldir, errors, tests, workingtree
from breezy.tests.script import TestCaseWithTransportAndScript
def test_make_with_trees(self):
    repo = self.make_repository('repo', shared=True)
    repo.set_make_working_trees(False)
    self.run_bzr('reconfigure --with-trees', working_dir='repo')
    self.assertIs(True, repo.make_working_trees())