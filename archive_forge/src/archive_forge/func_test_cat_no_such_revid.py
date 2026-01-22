from breezy.tests import TestCaseWithTransport
def test_cat_no_such_revid(self):
    tree = self.make_branch_and_tree('.')
    err = self.run_bzr('cat-revision abcd', retcode=3)[1]
    self.assertContainsRe(err, 'The repository .* contains no revision abcd.')