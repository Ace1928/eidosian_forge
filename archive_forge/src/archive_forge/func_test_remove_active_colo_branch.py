from breezy import controldir
from breezy.tests import TestCaseWithTransport
def test_remove_active_colo_branch(self):
    dir = self.make_repository('a').controldir
    branch = dir.create_branch('otherbranch')
    branch.create_checkout('a')
    self.run_bzr_error(['Branch is active. Use --force to remove it.\n'], 'rmbranch otherbranch -d %s' % branch.controldir.user_url)
    self.assertTrue(dir.has_branch('otherbranch'))
    self.run_bzr('rmbranch --force otherbranch -d %s' % branch.controldir.user_url)
    self.assertFalse(dir.has_branch('otherbranch'))