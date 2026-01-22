from breezy.tests import ChrootedTestCase, TestCaseWithTransport
def test_check_missing_branch_in_shared_repo(self):
    self.make_repository('shared', shared=True)
    out, err = self.run_bzr('check --branch shared')
    self.assertEqual(err, 'No branch found at specified location.\n')