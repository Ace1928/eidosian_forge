from breezy.tests import ChrootedTestCase, TestCaseWithTransport
def test_check_missing_partial(self):
    branch = self.make_branch('.')
    out, err = self.run_bzr('check --tree --branch')
    self.assertContainsRe(err, "Checking branch at '.*'\\.\\nNo working tree found at specified location\\.\\nchecked branch.*")