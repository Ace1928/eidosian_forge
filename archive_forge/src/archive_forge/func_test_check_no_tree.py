from breezy.tests import ChrootedTestCase, TestCaseWithTransport
def test_check_no_tree(self):
    self.make_branch('.')
    self.run_bzr('check')